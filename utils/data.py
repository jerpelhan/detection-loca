import os
import json
from typing import Tuple, List
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.ops import box_convert
from torchvision import transforms as T
from torchvision.transforms import functional as TVF


def tiling_augmentation(img, bboxes, density_map, resize, jitter, tile_size, hflip_p):

    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                if hflip[j, k] < hflip_p:
                    t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    tensors = [img, density_map]
    results = list()
    # whether to horizontally flip each tile
    hflip = torch.rand(num_tiles, num_tiles)

    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    img = resize(img[..., :int(y_tile*y_target), :int(x_tile*x_target)])

    density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
    density_map = density_map[..., :int(y_tile*y_target), :int(x_tile*x_target)]
    original_sum = density_map.sum()
    density_map = resize(density_map)
    density_map = density_map / density_map.sum() * original_sum

    if hflip[0, 0] < hflip_p:
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]  # TODO change
    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    return img, bboxes, density_map


class FSC147WithDensityMap(Dataset):

    def __init__(
        self, data_path, img_size, split='train', num_objects=3,
        tiling_p=0.5, zero_shot=False, skip_cars=False
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size))
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        with open(
            os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
            os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)
        if skip_cars:
            with open(
                os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), 'r'
            ) as file:
                classes = dict([
                    (line.strip().split()[0], ' '.join(line.strip().split()[1:]))
                    for line in file.readlines()
                ])
            print(len(self.image_names))
            self.image_names = [
                img_name for img_name in self.image_names if classes[img_name] != 'cars'
            ]
            print(len(self.image_names))
        if split == 'val' or split == 'test':
            self.labels = COCO(os.path.join(self.data_path, 'instances_'+split+'.json'))
            self.img_name_to_ori_id = self.map_img_name_to_ori_id()

    def __getitem__(self, idx: int) -> Tuple[Tensor, List[Tensor], Tensor]:
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            self.image_names[idx]
        ))
        w, h = img.size
        if self.split != 'train':
            img = T.Compose([
                T.ToTensor(),
                self.resize,
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                self.resize,
            ])(img)


        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size

        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)

        tiled = False
        if self.split == 'train' and torch.rand(1) < self.tiling_p:
            tiled = True
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img, bboxes, density_map = tiling_augmentation(
                img, bboxes, density_map, self.resize,
                self.jitter, tile_size, self.horizontal_flip_p
            )
        if self.split == 'train' and not tiled and torch.rand(1) < 0.0:
            tiled = True
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img_class = self.classes[self.image_names[idx]]
            candidate_images = [im for im, cl in self.classes.items() if cl != img_class]
            sampled = [candidate_images[j] for j in torch.randperm(len(candidate_images))[:3]]
            cutout_imgs = [img]
            for sampled_name in sampled:
                sampled_img = Image.open(os.path.join(
                    self.data_path,
                    'images_384_VarV2',
                    sampled_name
                ))
                sampled_img = T.Compose([
                    T.ToTensor(),
                    self.resize,
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(sampled_img)
                cutout_imgs.append(sampled_img)
            img, bboxes, density_map = cutout_augmentation(
                cutout_imgs, bboxes, density_map, self.resize,
                self.jitter, tile_size, self.horizontal_flip_p
            )

        if self.split == 'train':
            if not tiled:
                img = self.jitter(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if self.split == 'train' and not tiled and torch.rand(1) < self.horizontal_flip_p:
            img = TVF.hflip(img)
            density_map = TVF.hflip(density_map)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]

        return img, bboxes, density_map, idx

    def __len__(self):
        return len(self.image_names)

    def map_img_name_to_ori_id(self,):
        all_coco_imgs = self.labels.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id
    
    def get_gt_bboxes(self, idxs):
        if self.split == 'val' or self.split == 'test':
            l = []
            for i in idxs:
                img = Image.open(os.path.join(
                    self.data_path,
                    'images_384_VarV2',
                    self.image_names[i]
                ))
                w1, h1 = img.size
                coco_im_id = self.img_name_to_ori_id[self.image_names[i]]
                anno_ids = self.labels.getAnnIds([coco_im_id])
                annos = self.labels.loadAnns(anno_ids)
                box_centers = list()
                whs = list()
                xyxy_boxes = list()
                for anno in annos:
                    bbox = anno["bbox"]
                    x1, y1, w, h = bbox
                    box_centers.append([x1 + w / 2, y1 + h / 2])
                    whs.append([w, h])
                    xyxy_boxes.append([x1, y1, x1 + w, y1 + h])
                xyxy_boxes = np.array(xyxy_boxes, dtype=np.float32)
                xyxy_boxes = xyxy_boxes / torch.tensor([w1, h1, w1, h1]) * self.img_size
                l.append(xyxy_boxes)
            return l
