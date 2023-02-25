from models.cotr import build_model
from models.fcos import BoxList
from utils.data import FSC147WithDensityMap
from utils.arg_parser import get_argparser
from utils.losses import Criterion, Detection_criterion, calc_mAP

from time import perf_counter
import argparse
import os
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F, DataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

DATASETS = {
    'fsc147': FSC147WithDensityMap,
}


def reduce_dict(input_dict, average=False):
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def train(args):

    if args.skip_train:
        return

    # if 'SLURM_PROCID' in os.environ:
    #     world_size = int(os.environ['SLURM_NTASKS'])
    #     rank = int(os.environ['SLURM_PROCID'])
    #     gpu = rank % torch.cuda.device_count()
    #     print("Running on SLURM", world_size, rank, gpu)
    # else:
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     rank = int(os.environ['RANK'])
    #     gpu = int(os.environ['LOCAL_RANK'])
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # dist.init_process_group(
    #     backend='nccl', init_method='env://',
    #     world_size=world_size, rank=rank
    # )

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    # model = DistributedDataParallel(
    #     build_model(args).to(device),
    #     device_ids=[gpu],
    #     output_device=gpu
    # )
    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    backbone_params = dict()
    non_backbone_params = dict()
    fcos_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
        elif 'fcos' in n:
            fcos_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            # {'params': non_backbone_params.values()},
            # {'params': backbone_params.values(), 'lr': args.backbone_lr}
            {'params': fcos_params.values()}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best = 10000000000000
        best_mAP = 0

    criterion = Criterion(args)
    aux_criterion = Criterion(args, aux=True)
    det_criterion = Detection_criterion( 
            [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]], #config.sizes,
            2.0,    #config.gamma,
            0.25, #config.alpha,
            'giou', #config.iou_loss_type,
            True, #config.center_sample,
            [1],#config.fpn_strides,
            1.5, #config.pos_radius,
        )

    train = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot or args.orig_dmaps,
        skip_cars=args.skip_cars
    )
    val = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot or args.orig_dmaps,
    )
    train_loader = DataLoader(
        train,
        # sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val,
        # sampler=DistributedSampler(val),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0
    )
    print("NUM STEPS", len(train_loader) * args.epochs)
    # print(rank, len(train_loader))
    for epoch in range(start_epoch + 1, args.epochs + 1):
        start = perf_counter()
        train_losses = {k: torch.tensor(0.0).to(device) for k in criterion.losses.keys()}
        val_losses = {k: torch.tensor(0.0).to(device) for k in criterion.losses.keys()}
        aux_train_losses = {k: torch.tensor(0.0).to(device) for k in aux_criterion.losses.keys()}
        aux_val_losses = {k: torch.tensor(0.0).to(device) for k in aux_criterion.losses.keys()}
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        model.train()
        for img, bboxes, density_map, ids in train_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            targets = BoxList(bboxes/args.image_size*args.fcos_pred_size, args.fcos_pred_size, mode='xyxy').to(device)
            targets.fields['labels'] =  [1 for i in range(args.batch_size)]
            optimizer.zero_grad()
            cls_pred, box_pred, center_pred, location, out, aux_out = model(img, bboxes, density_map, targets)
            if args.normalized_l2:
                with torch.no_grad():
                    num_objects = density_map.sum()
                    dist.all_reduce_multigpu([num_objects])
            else:
                num_objects = None

            main_losses = criterion(out, density_map, bboxes, num_objects)
            aux_losses = [
                aux_criterion(aux, density_map, bboxes, num_objects) for aux in aux_out
            ]
            det_losses = det_criterion(location, cls_pred, box_pred, center_pred, targets)
            loss = (
                sum([ml for ml in main_losses.values()]) +
                sum([al for alls in aux_losses for al in alls.values()]) +
                sum([dl for dl in det_losses.values()]) * 100
            )
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_losses = {
                k: train_losses[k] + main_losses[k] * img.size(0) for k in train_losses.keys()
            }
            aux_train_losses = {
                k: aux_train_losses[k] + sum([a[k] for a in aux_losses]) * img.size(0)
                for k in aux_train_losses.keys()
            }
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()

        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map, ids in val_loader:
                gt_bboxes = val.get_gt_bboxes(ids)
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)
                target = BoxList(bboxes/args.image_size*args.fcos_pred_size, args.fcos_pred_size, mode='xyxy').to(device)
                target.fields['labels'] = [1 for i in range(args.batch_size)]
                optimizer.zero_grad()
                cls_pred, box_pred, center_pred, location, out, aux_out = model(img, bboxes, density_map, target)
                bboxes = model.module.postprocessor(
                    location, cls_pred, box_pred, center_pred, [[args.fcos_pred_size,args.fcos_pred_size] for i in range(args.batch_size)], [T.Resize((64,64))(out)])
                mAP = calc_mAP(bboxes, gt_bboxes).to(device)
                if args.normalized_l2:
                    with torch.no_grad():
                        num_objects = density_map.sum()
                        dist.all_reduce_multigpu([num_objects])
                else:
                    num_objects = None
                main_losses = criterion(out, density_map, bboxes, num_objects)
                aux_losses = [
                    aux_criterion(aux, density_map, bboxes, num_objects) for aux in aux_out
                ]
                loss = (
                    sum([ml for ml in main_losses.values()]) +
                    sum([al for alls in aux_losses for al in alls.values()])
                )
                val_losses = {
                    k: val_losses[k] + main_losses[k] * img.size(0) for k in val_losses.keys()
                }
                aux_val_losses = {
                    k: aux_val_losses[k] + sum([a[k] for a in aux_losses]) * img.size(0)
                    for k in aux_val_losses.keys()
                }
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()

        # train_losses = reduce_dict(train_losses)
        # val_losses = reduce_dict(val_losses)
        # aux_train_losses = reduce_dict(aux_train_losses)
        # aux_val_losses = reduce_dict(aux_val_losses)
        # dist.all_reduce_multigpu([train_ae])
        # dist.all_reduce_multigpu([val_ae])
        # dist.all_reduce_multigpu([mAP])

        scheduler.step()

        # if rank == 0:
        end = perf_counter()
        best_epoch = False
        # if val_ae.item() / len(val) < best:
        #     best = val_ae.item() / len(val)
        # if mAP > best:
        if mAP > best_mAP:
            best_mAP = mAP
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_ae': val_ae.item() / len(val)
            }
            torch.save(
                checkpoint,
                os.path.join(args.model_path, f'{args.model_name}.pth')
            )
            best_epoch = True

        print("Epoch", epoch)
        print("mAP", mAP)
        print({k: v.item() / len(train) for k, v in train_losses.items()})
        print({k: v.item() / len(val) for k, v in val_losses.items()})
        print({k: v.item() / len(train) for k, v in aux_train_losses.items()})
        print({k: v.item() / len(val) for k, v in aux_val_losses.items()})
        print(
            train_ae.item() / len(train),
            val_ae.item() / len(val),
            end - start,
            'best' if best_epoch else '',
        )

    if args.skip_test:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(args):

    if args.skip_test:
        return

    if not args.skip_train:
        dist.barrier()
    #
    # if 'SLURM_PROCID' in os.environ:
    #     world_size = int(os.environ['SLURM_NTASKS'])
    #     rank = int(os.environ['SLURM_PROCID'])
    #     gpu = rank % torch.cuda.device_count()
    #     print("Running on SLURM", world_size, rank, gpu)
    # else:
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     rank = int(os.environ['RANK'])
    #     gpu = int(os.environ['LOCAL_RANK'])
    gpu=0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # if args.skip_train:
    #     dist.init_process_group(
    #         backend='nccl', init_method='env://',
    #         world_size=world_size, rank=rank
    #     )

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    # model.load_state_dict(
    #     torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'))['model']
    # )

    # TODO remove, just for test
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'))['model'], strict=False
    )

    for split in ['val', 'test']:
        test = DATASETS[args.dataset](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            zero_shot=args.zero_shot or args.orig_dmaps
        )
        test_loader = DataLoader(
            test,
            # sampler=DistributedSampler(test),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=0
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()
        for img, bboxes, density_map, ids in test_loader:
            gt_bboxes = test.get_gt_bboxes(ids)
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            target = BoxList(bboxes/args.image_size*args.fcos_pred_size, args.fcos_pred_size, mode='xyxy').to(device)
            target.fields['labels'] = [1,1,1]
            cls_pred, box_pred, center_pred, location, out, aux_out = model(img, bboxes, density_map, target)
            bboxes = model.module.postprocessor(
                    location, cls_pred, box_pred, center_pred, [[args.fcos_pred_size,args.fcos_pred_size] for i in range(args.batch_size)], [T.Resize((64,64))(density_map)])
            mAP = calc_mAP(bboxes, gt_bboxes).to(device)
            from matplotlib import pyplot as plt
            import skimage
            a = skimage.feature.peak_local_max(np.array(T.Resize((64, 64))(density_map)[0][0].cpu()), exclude_border=0)
            boxes = []
            t, b, l, r = box_pred[0][0]
            for x, y in a:
                boxes.append([y - b[x][y].item(), x - t[x][y].item(), y + r[x][y].item(), x + l[x][y].item()])
                # ab = [[y - b[x][y].item(), x - t[x][y].item(), y + r[x][y].item(), x + l[x][y].item()]
                #  for x in range(max(x1-args.nms_radi,0),min(args.fcos_pred_size,x1+args.nms_radi)) for y in
                #       range(max(y1-args.nms_radi,0),min(args.fcos_pred_size,y1+args.nms_radi))]


            img_ = np.array(img.cpu()[0].permute(1, 2, 0))
            img_ = img_ - np.min(img_)
            img_ = img_ / np.max(img_)

            fig, (ax0, ax2) = plt.subplots(1, 2)
            ax0.imshow(np.array(img_))
            ax0.set_title("Img")
            # ax1.imshow(np.array(img_))
            # ax1.imshow(np.array(density_map.cpu()[0][0]), alpha=0.8)
            # ax1.set_title("GT")
            ax2.imshow(np.array(img_))
            ax2.imshow(np.array(density_map.cpu()[0][0]), alpha=0.8
                       )
            bboxes_ = np.array(boxes) * 8
            for i in range(len(bboxes_)):
                ax2.plot([bboxes_[i][0], bboxes_[i][0], bboxes_[i][2], bboxes_[i][2], bboxes_[i][0]],
                         [bboxes_[i][1], bboxes_[i][3], bboxes_[i][3], bboxes_[i][1], bboxes_[i][1]])
            ax2.set_title("Prediction")
            plt.show()
            ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()
            se += ((
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ) ** 2).sum()

        dist.all_reduce_multigpu([ae])
        dist.all_reduce_multigpu([se])
        dist.all_reduce_multigpu([mAP])

        print(
            f"{split} set",
            f"MAE {ae.item() / len(test)} RMSE {torch.sqrt(se / len(test)).item()}",
        )
        print("mAP", mAP)

    dist.destroy_process_group()


if __name__ == '__main__':
    print("HERE", torch.cuda.device_count())
    parser = argparse.ArgumentParser('DMAPTR', parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    train(args)
    evaluate(args)
