import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import ops





class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale



class FCOSHead(nn.Module):
    def __init__(self, in_channels, n_class, n_conv, prior):
        super().__init__()
        conv_channels=256
        n_class = n_class - 1

        cls_tower = []
        bbox_tower = []

        # 1x1 convolutional pooling for dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1)

        for i in range(n_conv):
            cls_tower.append(
                nn.Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, conv_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, conv_channels))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_pred = nn.Conv2d(conv_channels, n_class, 3, padding=1)
        self.bbox_pred = nn.Conv2d(conv_channels, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(conv_channels, 1, 3, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_pred, self.bbox_pred,
                        self.center_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)


        prior_bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, prior_bias)

        # self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])
        self.scale = Scale(1.0)

    def forward(self, x):
        logits = []
        bboxes = []
        centers = []

        x = self.conv1(x)

        # for feat, scale in zip(x, self.scales):
        cls_out = self.cls_tower(x)

        logits.append(self.cls_pred(cls_out))
        centers.append(self.center_pred(cls_out))

        bbox_out = self.bbox_tower(x)
        bbox_out = torch.exp(self.scale(self.bbox_pred(bbox_out)))

        bboxes.append(bbox_out)

        return logits, bboxes, centers


    
FLIP_LEFT_RIGHT = 0

class FCOSPostprocessor(nn.Module):
    def __init__(self, threshold, top_n, nms_threshold, post_top_n, min_size, n_class):
        super().__init__()

        self.threshold = threshold
        self.top_n = top_n
        self.nms_threshold = nms_threshold
        self.post_top_n = post_top_n
        self.min_size = min_size
        self.n_class = n_class

    def forward_single_feature_map(
        self, location, cls_pred, box_pred, center_pred, image_sizes, density_map
    ):
        batch, channel, height, width = cls_pred.shape

        cls_pred = cls_pred.view(batch, channel, height, width).permute(0, 2, 3, 1)
        cls_pred = cls_pred.reshape(batch, -1, channel).sigmoid()

        box_pred = box_pred.view(batch, 4, height, width).permute(0, 2, 3, 1)
        box_pred = box_pred.reshape(batch, -1, 4)

        center_pred = center_pred.view(batch, 1, height, width).permute(0, 2, 3, 1)
        center_pred = center_pred.reshape(batch, -1).sigmoid()
        candid_ids = cls_pred > 0.07
        top_ns = candid_ids.view(batch, -1).sum(1)
        # candid_ids = cls_pred > 0.5
        # top_ns = candid_ids.view(batch, -1).sum(1)
        top_ns = top_ns.clamp(max=self.top_n)

        cls_pred = cls_pred * center_pred[:, :, None]

        density_map = density_map.view(batch, channel, height, width).permute(0, 2, 3, 1)
        density_map = density_map.reshape(batch, -1, channel).sigmoid()
        candid_ids = density_map > 0.0001
        density_map = density_map.reshape(batch, -1, channel)
        ids = density_map <= 0.0001
        density_map[ids] = 0
        cls_pred = density_map
        # top_ns = candid_ids.view(batch, -1).sum(1)
        # top_ns = top_ns.clamp(max=self.top_n)

        results = []

        for i in range(batch):
            cls_p = cls_pred[i]
            candid_id = candid_ids[i]
            cls_p = cls_p[candid_id]
            candid_nonzero = candid_id.nonzero()
            box_loc = candid_nonzero[:, 0]
            class_id = candid_nonzero[:, 1] + 1

            box_p = box_pred[i]
            box_p = box_p[box_loc]
            loc = location[box_loc]

            top_n = top_ns[i]

            if candid_id.sum().item() > top_n.item():
                cls_p, top_k_id = cls_p.topk(top_n, sorted=False)
                class_id = class_id[top_k_id]
                box_p = box_p[top_k_id]
                loc = loc[top_k_id]

            detections = torch.stack(
                [
                    loc[:, 0] - box_p[:, 0],
                    loc[:, 1] - box_p[:, 1],
                    loc[:, 0] + box_p[:, 2],
                    loc[:, 1] + box_p[:, 3],
                ],
                1,
            )

            height, width = image_sizes[i]

            boxlist = BoxList(detections, (int(width), int(height)), mode='xyxy')
            boxlist.fields['labels'] = class_id
            boxlist.fields['scores'] = torch.sqrt(cls_p)
            boxlist = boxlist.clip(remove_empty=False)
            boxlist = remove_small_box(boxlist, self.min_size)

            results.append(boxlist)

        return results

    def forward(self, location, cls_pred, box_pred, center_pred, image_sizes, density_maps):
        boxes = []

        for loc, cls_p, box_p, center_p, density_map in zip(
            location, cls_pred, box_pred, center_pred, density_maps
        ):
            boxes.append(
                self.forward_single_feature_map(
                    loc, cls_p, box_p, center_p, image_sizes, density_map
                )
            )

        boxlists = list(zip(*boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_scales(boxlists)

        return boxlists

    def select_over_scales(self, boxlists):
        results = []

        for boxlist in boxlists:
            scores = boxlist.fields['scores']
            labels = boxlist.fields['labels']
            box = boxlist.box

            result = []

            for j in range(1, self.n_class):
                id = (labels == j).nonzero().view(-1)
                score_j = scores[id]
                box_j = box[id, :].view(-1, 4)
                box_by_class = BoxList(box_j, boxlist.size, mode='xyxy')
                box_by_class.fields['scores'] = score_j
                box_by_class = boxlist_nms(box_by_class, score_j, self.nms_threshold)
                n_label = len(box_by_class)
                box_by_class.fields['labels'] = torch.full(
                    (n_label,), j, dtype=torch.int64, device=scores.device
                )
                result.append(box_by_class)

            result = cat_boxlist(result)
            n_detection = len(result)

            # if n_detection > self.post_top_n > 0:
            #     scores = result.fields['scores']
            #     img_threshold, _ = torch.kthvalue(
            #         scores.cpu(), n_detection - self.post_top_n + 1
            #     )
            #     keep = scores >= img_threshold.item()
            #     keep = torch.nonzero(keep).squeeze(1)
            #     result = result[keep]

            results.append(result)

        return results
FLIP_TOP_BOTTOM = 1


class BoxList:
    def __init__(self, box, image_size, mode='xyxy'):
        device = box.device if hasattr(box, 'device') else 'cpu'
        box = torch.as_tensor(box, dtype=torch.float32, device=device)

        self.box = box
        self.size = image_size
        self.mode = mode

        self.fields = {}

    def convert(self, mode):
        if mode == self.mode:
            return self

        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if mode == 'xyxy':
            box = torch.cat([x_min, y_min, x_max, y_max], -1)
            box = BoxList(box, self.size, mode=mode)

        elif mode == 'xywh':
            remove = 1
            box = torch.cat(
                [x_min, y_min, x_max - x_min + remove, y_max - y_min + remove], -1
            )
            box = BoxList(box, self.size, mode=mode)

        box.copy_field(self)

        return box

    def copy_field(self, box):
        for k, v in box.fields.items():
            self.fields[k] = v

    def area(self):
        box = self.box

        if self.mode == 'xyxy':
            remove = 1

            area = (box[:, 2] - box[:, 0] + remove) * (box[:, 3] - box[:, 1] + remove)

        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]

        return area

    def split_to_xyxy(self):
        if self.mode == 'xyxy':
            x_min, y_min, x_max, y_max = self.box.split(1, dim=-1)

            return x_min, y_min, x_max, y_max

        elif self.mode == 'xywh':
            remove = 1
            x_min, y_min, w, h = self.box.split(1, dim=-1)

            return (
                x_min,
                y_min,
                x_min + (w - remove).clamp(min=0),
                y_min + (h - remove).clamp(min=0),
            )

    def __len__(self):
        return self.box.shape[0]

    def __getitem__(self, index):
        box = BoxList(self.box[index], self.size, self.mode)

        for k, v in self.fields.items():
            box.fields[k] = v[index]

        return box

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled = self.box * ratio
            box = BoxList(scaled, size, mode=self.mode)

            for k, v in self.fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)

                box.fields[k] = v

            return box

        ratio_w, ratio_h = ratios
        x_min, y_min, x_max, y_max = self.split_to_xyxy()
        scaled_x_min = x_min * ratio_w
        scaled_x_max = x_max * ratio_w
        scaled_y_min = y_min * ratio_h
        scaled_y_max = y_max * ratio_h
        scaled = torch.cat([scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max], -1)
        box = BoxList(scaled, size, mode='xyxy')

        for k, v in self.fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)

            box.fields[k] = v

        return box.convert(self.mode)

    def transpose(self, method):
        width, height = self.size
        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if method == FLIP_LEFT_RIGHT:
            remove = 1

            transpose_x_min = width - x_max - remove
            transpose_x_max = width - x_min - remove
            transpose_y_min = y_min
            transpose_y_max = y_max

        elif method == FLIP_TOP_BOTTOM:
            transpose_x_min = x_min
            transpose_x_max = x_max
            transpose_y_min = height - y_max
            transpose_y_max = height - y_min

        transpose_box = torch.cat(
            [transpose_x_min, transpose_y_min, transpose_x_max, transpose_y_max], -1
        )
        box = BoxList(transpose_box, self.size, mode='xyxy')

        for k, v in self.fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)

            box.fields[k] = v

        return box.convert(self.mode)

    def clip(self, remove_empty=True):
        remove = 1

        max_width = self.size[0] - remove
        max_height = self.size[1] - remove

        self.box[:, 0].clamp_(min=0, max=max_width)
        self.box[:, 1].clamp_(min=0, max=max_height)
        self.box[:, 2].clamp_(min=0, max=max_width)
        self.box[:, 3].clamp_(min=0, max=max_height)

        if remove_empty:
            box = self.box
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])

            return self[keep]

        else:
            return self

    def to(self, device):
        box = BoxList(self.box.to(device), self.size, self.mode)

        for k, v in self.fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)

            box.fields[k] = v

        return box


def remove_small_box(boxlist, min_size):
    box = boxlist.convert('xywh').box
    _, _, w, h = box.unbind(dim=1)
    keep = (w >= min_size) & (h >= min_size)
    keep = keep.nonzero().squeeze(1)

    return boxlist[keep]


def cat_boxlist(boxlists):
    size = boxlists[0].size
    mode = boxlists[0].mode
    field_keys = boxlists[0].fields.keys()

    box_cat = torch.cat([boxlist.box for boxlist in boxlists], 0)
    new_boxlist = BoxList(box_cat, size, mode)

    for field in field_keys:
        data = torch.cat([boxlist.fields[field] for boxlist in boxlists], 0)
        new_boxlist.fields[field] = data

    return new_boxlist


def boxlist_nms(boxlist, scores, threshold, max_proposal=-1):
    if threshold <= 0:
        return boxlist

    mode = boxlist.mode
    boxlist = boxlist.convert('xyxy')
    box = boxlist.box
    keep = ops.nms(box, scores, threshold)

    if max_proposal > 0:
        keep = keep[:max_proposal]

    boxlist = boxlist[keep]

    return boxlist.convert(mode)