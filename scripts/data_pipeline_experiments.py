from torch.utils.data import DataLoader

import sys
sys.path.append("/usr/stud/cad/projects/VisTR/VisTR/")
from datasets import build_dataset, get_coco_api_from_dataset

"""
YoutubeVIS data loader
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision

import tqdm
sys.path.append("/usr/stud/cad/projects/VisTR/cocoapi/PythonAPI")
from pycocotools_.ytvos import YTVOS
from pycocotools_.ytvoseval import YTVOSeval
import datasets.transforms as T
from datasets.ytvos import YTVOSDataset
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
from datasets.ytvos import make_coco_transforms, make_custom_transforms, ConvertCocoPolysToMask, make_default_transform
from scripts.viz_utils_transforms import visualize_transformed_images
import numpy as np
from timeit import default_timer as timer
from util.misc import nested_dict_to_namespace, collate_fn, init_distributed_mode


# class SpeedTestYTVOSDataset(YTVOSDataset):
#     def __init__(self, img_folder, ann_file, return_masks, num_frames):
#         super().__init__(img_folder, ann_file, None, return_masks, num_frames, load_image_func=None, prepare_targets=None)
#         self.prepare = None
#
#     def get_most_annot_idx(self, targets_prepare):
#         most_num_annots = 0
#         highest_idx = 0
#         for idx in tqdm.tqdm(range(len(self.img_ids))):
#             vid, frame_id = self.img_ids[idx]
#             vid_id = self.vid_infos[vid]['id']
#             img = []
#             vid_len = len(self.vid_infos[vid]['file_names'])
#             inds = list(range(self.num_frames))
#             inds = [i % vid_len for i in inds]  # Tecnica repetir desde el començament el video si es mes llarg + girar llista
#             inds = inds[::-1]
#             ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
#             target = self.ytvos.loadAnns(ann_ids)
#             target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
#             target = targets_prepare(img, target, inds, self.num_frames)
#             num_annots = target["boxes"].shape[0]
#             if num_annots > most_num_annots:
#                 highest_idx = idx
#                 most_num_annots = num_annots
#         print(f"Highest idx {highest_idx}  NUM OBJ: {most_num_annots}")
#         return highest_idx


    # def get_item_standard(self, idx, transform_pipeline, targets_prepare, use_opencv=False):
    #     vid, frame_id = self.img_ids[idx]
    #     vid_id = self.vid_infos[vid]['id']
    #     img = []
    #     vid_len = len(self.vid_infos[vid]['file_names'])
    #     inds = list(range(self.num_frames))
    #     inds = [i % vid_len for i in inds]  # Tecnica repetir desde el començament el video si es mes llarg + girar llista
    #     inds = inds[::-1]
    #     for j in range(self.num_frames):
    #         img_path = os.path.join(str(self.img_folder),
    #                                 self.vid_infos[vid]['file_names'][frame_id - inds[j]])  # Samplejar indices del video pero centrat a frame_id
    #         if use_opencv:
    #             img.append(cv2.imread(img_path))
    #         else:
    #             img.append(Image.open(img_path).convert('RGB'))
    #     video_name = img_path.split("/")[-2]
    #     ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
    #     target = self.ytvos.loadAnns(ann_ids)
    #     target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
    #     target = targets_prepare(img[0], target, inds, self.num_frames)
    #     img, target = transform_pipeline(img, target)
    #     if isinstance(img, list):
    #         img = torch.stack(img, dim=0)
    #
    #     return img, target, video_name


# def build(num_frames):
#     root = Path("/usr/prakt/p028/data")
#     assert root.exists(), f'provided Data path {root} does not exist'
#
#     PATHS = {
#         "yt_vis_train": ((root / "Youtube_VIS/train/JPEGImages", root / "Youtube_VIS/train/" / 'train.json'), 40),
#         "ovis_train": ((root / "OVIS/train/", root / "OVIS/" / "annotations_train.json"), 25),
#     }
#     img_folder, ann_file = PATHS["ovis_train"][0]
#     dataset = SpeedTestYTVOSDataset(img_folder, ann_file, return_masks=True, num_frames=num_frames)
#     cat_names = {cat_id: cat_info["name"] for cat_id, cat_info in dataset.ytvos.cats.items()}
#     cat_names[0] = "Bkg"
#
#     return dataset, cat_names


if __name__ == "__main__":
    # idx_to_test = [33133] * 100
    # idx_to_test = [10] * 5
    out_path = "/usr/stud/cad/results/DefVisTr/TransformsPipeline"


    args = {
        "dataset_type": "vis",
        "data_path": "/usr/stud/cad/p028/data",
        # "transform_pipeline": "default",
        "transform_pipeline": "optimized_transform",
        "train_set": "train_train_val_split",
        "masks": True,
        "num_frames": 36,
        "num_workers": 0,
        "batch_size": 1,
        "distributed": False,
    }

    args = nested_dict_to_namespace(args)

    # init_distributed_mode(args)
    train_dataset, num_classes = build_dataset(image_set="train", args=args)


    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(train_dataset)

    else:
        sampler_train = torch.utils.data.SequentialSampler(train_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers)

    # idx_to_test = []
    # for i in range(0, 100):
    #     idx_to_test.append(random.randint(0, len(train_dataset)))
    # cat_names = {id_: info["name"] for id_, info in train_dataset.ytvos.cats.items()}
    # times = []
    # idx_to_test = [3727]

    start = timer()
    for sample, target in data_loader_train:
        # start = timer()
        print(f"ID: {target[0]['image_id'] }")
        # images, targets = train_dataset[i]
        #
        # # images, targets, video_name = dataset.get_item_standard(i, default_transform, default_prepare)
        # # images_albu, targets_albu, video_name_albu = dataset.get_item_standard(i, albu_transforms, valued_mask_prepare_numpy, True)
        # end = timer()
        # times.append(end-start)
        # visualize_transformed_images(images, targets, os.path.join(out_path, str(i), f"default_transforms_{i}"), cat_names)
        #
        # # images, targets, video_name = dataset.get_item_standard(i, albu_transforms, valued_mask_prepare_numpy, True)
        # # visualize_transformed_images(images, targets, os.path.join(out_path, str(i), f"albu_transforms_{i}"), cat_names)
        #
        # print("Frame finish")
        # _ = dataset.get_item_standard(i, default_transform, default_prepare)
        # _ = dataset.get_item_standard(i, no_resizes_no_crop_transform, valued_mask_prepare)


    mean_time = np.mean(times)
    std = np.std(times)
    print(f"Mean time {mean_time}, STD {std}")