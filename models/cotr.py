from models.fcos import FCOSHead, FCOSPostprocessor
from .backbone import Backbone
from .transformer import TransformerEncoder, TransformerDecoder
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor

import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
import torchvision.transforms as T


class COTR(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        swav_backbone: bool,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        use_query_pos_emb: bool,
        zero_shot: bool,
        use_objectness: bool,
        use_appearance: bool
    ):

        super(COTR, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.use_query_pos_emb = use_query_pos_emb
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_objectness = use_objectness
        self.use_appearance = use_appearance

        self.backbone = Backbone(
            backbone_name, pretrained=True, dilation=False, reduction=reduction,
            swav=swav_backbone, requires_grad=train_backbone
        )
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )

        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        if num_decoder_layers > 0:
            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers, emb_dim=emb_dim, num_heads=num_heads,
                dropout=0, layer_norm_eps=layer_norm_eps,
                mlp_factor=mlp_factor, norm_first=norm_first,
                activation=activation, norm=norm,
                attn1=not self.zero_shot and self.use_appearance
            )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction) for _ in range(num_decoder_layers - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

        if self.use_objectness:
            if not self.zero_shot:
                self.objectness = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.ReLU(),
                    nn.Linear(64, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, self.kernel_dim**2 * emb_dim)
                )
            else:
                self.objectness = nn.Parameter(
                    torch.empty((self.num_objects, self.kernel_dim**2, emb_dim))
                )
                nn.init.normal_(self.objectness)
        
        self.fcos = FCOSHead(
            7168, 2, 4, 0.01
        )

        self.postprocessor = FCOSPostprocessor(
            0.15,  # config.threshold,
            1000,  # config.top_n,
            0.2,  # config.nms_threshold,
            100,  # config.post_top_n,
            0,  # config.min_size,
            2  # config.n_class,
        )

    def compute_location(self, features):
        locations = []

        # for i, feat in enumerate(features):
        _, _, height, width = features.shape
        location_per_level = self.compute_location_per_level(
            height, width, 1, features.device
        )
        locations.append(location_per_level)

        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2

        return location

    def forward(self, x, bboxes, gt_density, targets):
        # backbone
        backbone_features = self.backbone(x)
        _, _, bb_h, bb_w = backbone_features.size()
        # prepare the encoder input
        src = self.input_proj(backbone_features)
        bs, c, h, w = src.size()
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)
        src = src.flatten(2).permute(2, 0, 1)

        # push through the encoder
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
        else:
            memory = src

        # prepare the decoder input
        x = memory.permute(1, 2, 0).reshape(-1, self.emb_dim, bb_h, bb_w)

        # extract the objectness
        if self.use_objectness and not self.zero_shot:
            box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
            box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
            box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]
            objectness = self.objectness(box_hw).reshape(
                bs, -1, self.kernel_dim ** 2, self.emb_dim
            ).flatten(1, 2).transpose(0, 1)
        elif self.zero_shot:
            objectness = self.objectness.expand(bs, -1, -1, -1).flatten(1, 2).transpose(0, 1)
        else:
            objectness = None

        # if not zero shot add appearance
        if not self.zero_shot and self.use_appearance:
            # reshape bboxes into the format suitable for roi_align
            bboxes = torch.cat([
                torch.arange(
                    bs, requires_grad=False
                ).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
                bboxes.flatten(0, 1),
            ], dim=1)
            appearance = roi_align(
                x,
                boxes=bboxes, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction, aligned=True
            ).permute(0, 2, 3, 1).reshape(
                bs, self.num_objects * self.kernel_dim ** 2, -1
            ).transpose(0, 1)
        else:
            appearance = None

        if self.use_query_pos_emb:
            query_pos_emb = self.pos_emb(
                bs, self.kernel_dim, self.kernel_dim, src.device
            ).flatten(2).permute(2, 0, 1).repeat(self.num_objects, 1, 1)
        else:
            query_pos_emb = None

        if self.num_decoder_layers > 0:
            weights = self.decoder(
                objectness if objectness is not None else appearance,
                appearance, memory, pos_emb, query_pos_emb
            )
        else:
            if objectness is not None and appearance is not None:
                weights = (objectness + appearance).unsqueeze(0)
            else:
                weights = (objectness if objectness is not None else appearance).unsqueeze(0)

        # prepare regression decoder input
        x = memory.permute(1, 2, 0).reshape(-1, self.emb_dim, bb_h, bb_w)

        outputs = list()
        for i in range(weights.size(0)):
            kernels = weights[i, ...].permute(1, 0, 2).reshape(
                bs, self.num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]
            correlation_maps = F.conv2d(
                torch.cat([x for _ in range(self.num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                kernels,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=kernels.size(0)
            ).view(
                bs, self.num_objects, self.emb_dim, bb_h, bb_w
            )
            softmaxed_correlation_maps = correlation_maps.softmax(dim=1)
            correlation_maps = torch.mul(softmaxed_correlation_maps, correlation_maps).sum(dim=1)

            if i == weights.size(0) - 1:
                # send through regression head
                _x = self.regression_head(correlation_maps)
            else:
                _x = self.aux_heads[i](correlation_maps)
            outputs.append(_x)

        densityGT_resized= T.Resize((backbone_features.size()[2],backbone_features.size()[3]))(gt_density)
        densityGT_resized = torch.cat([densityGT_resized for _ in range(3584)], dim=1)
        b = torch.cat((densityGT_resized, backbone_features), dim=1)
        cls_pred, box_pred, center_pred = self.fcos(b)

        location = self.compute_location(b)

        # loss_cls, loss_box, loss_center = self.loss(
        #         location, cls_pred, box_pred, center_pred, targets
        #     )
        # losses = {
        #     'loss_box': loss_box,
        #     'loss_cls': loss_cls,
        #     'loss_center': loss_center,
        # }
        # bboxes = self.postprocessor(
        #         location, cls_pred, box_pred, center_pred, [[64,64],[64,64],[64,64]]
        #     )

        return cls_pred, box_pred, center_pred, location, outputs[-1], outputs[:-1]
        # return bboxes, losses, outputs[-1], outputs[:-1]


def build_model(args):
    return COTR(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_decoder_layers=args.num_dec_layers,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
        use_query_pos_emb=args.use_query_pos_emb,
        use_objectness=args.use_objectness,
        use_appearance=args.use_appearance
    )
