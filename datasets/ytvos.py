"""
YoutubeVIS data loader
"""
import math
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import sys
sys.path.append("/usr/stud/cad/projects/VisTR/cocoapi/PythonAPI")
from pycocotools_.ytvos import YTVOS
from pycocotools_.ytvoseval import YTVOSeval
import datasets.transforms as T

import torchvision.transforms as torch_T
import json
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
from .optimized_transforms import ClipTransformsApplier, ConvertCocoPolysToValuedMaskNumpy
from .transforms import ConvertCocoPolysToMask, make_coco_transforms

class YTVOSDataset:
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames, load_image_func, prepare_targets, temporal_coherence,
                 reversed_sampling, focal_loss, use_non_valid_class):

        self.use_non_valid_class = use_non_valid_class
        self.img_folder = img_folder
        self.temporal_coherence = temporal_coherence
        self.reversed_sampling = reversed_sampling
        self.focal_loss = focal_loss
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.load_image_func = load_image_func
        self.prepare = prepare_targets
        self.ytvos = YTVOS(ann_file)
        # all_ = {}
        # for _, info in self.ytvos.anns.items():
        #     if info["video_id"] not in all_:
        #         all_[info["video_id"]] = 0
        #
        #     all_[info["video_id"]] += 1
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        if self.temporal_coherence:
            for idx, vid_info in enumerate(self.vid_infos):
                if vid_info["length"] < self.num_frames:
                    # Length video shorter than num_frames: We introduce padding as we do not want to ignore this clip
                    self.img_ids.append((idx, 0))
                    continue
                for frame_id in range(len(vid_info['filenames'])):
                    if frame_id + self.num_frames <= vid_info["length"]:
                        self.img_ids.append((idx, frame_id))
                    else:
                        break
        else:
            for idx, vid_info in enumerate(self.vid_infos):
                for frame_id in range(len(vid_info['filenames'])):
                    self.img_ids.append((idx, frame_id))


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        vid, frame_id = self.img_ids[idx]
        vid_id = self.vid_infos[vid]['id']
        img = []
        if self.temporal_coherence or self.reversed_sampling:
            vid_len = self.vid_infos[vid]['length']
            inds = list(range(0, - (vid_len-frame_id - 1), -1))
            if len(inds) >= self.num_frames:
                inds = inds[:self.num_frames]
            else:
                max_timestep = vid_len - frame_id - 1
                min_timestep = - frame_id
                list1 = list(range(-max_timestep, -min_timestep, 1))
                list2 = list(range(-min_timestep, -max_timestep, -1))
                while len(inds) < self.num_frames:
                    inds.extend(list1+list2)
                inds = inds[:self.num_frames]
        else:
            vid_len = len(self.vid_infos[vid]['file_names'])
            inds = list(range(self.num_frames))
            inds = [i % vid_len for i in inds][::-1]

        # picked_frames = [self.vid_infos[vid]['file_names'][frame_id - inds[j]] for j in range(self.num_frames)]
        for j in range(self.num_frames):
            img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id - inds[j]])
            img.append(self.load_image_func(img_path))
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
        target = self.prepare(img[0], target, inds, self.num_frames)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        target["num_trajectories"] = torch.tensor(target["labels"].shape[0] // self.num_frames)

        # VisTr implementation sets non_valid targets class to 0 and background to Num_classes, resulting in an extra category.
        # We have changed this to have non_valid target class mapped to background class.
        target["labels"] = target["labels"] - 1
        # Background is set to class 40
        num_cats = self.cat_ids[-1]
        for idx in range(target["labels"].shape[0]):
            if target["labels"][idx] == -1:
                target["labels"][idx] = num_cats

        if isinstance(img, list):
            img = torch.cat(img, dim=0)

        return img, target


class YTVOSVideoClip(torch.utils.data.dataset.Dataset):

    def __init__(self, images_folder, video_id, video_clips, original_size, last_real_idx, real_video_length, transform, final_video_length, cat_names):
        self.video_id = video_id
        self.video_clips = video_clips
        self.last_real_idx = last_real_idx
        self.real_video_length = real_video_length
        self.images_folder = images_folder
        self.transform = transform
        self.original_size = original_size
        self.final_video_length = final_video_length
        self.cat_names = cat_names

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, item):
        video_clip = self.video_clips[item]
        clip_imgs_set = []
        for k in range(len(video_clip)):
            im = Image.open(os.path.join(self.images_folder, video_clip[k]))
            clip_imgs_set.append(self.transform(im).unsqueeze(0))
        img = torch.cat(clip_imgs_set, 0)
        return img

class YTVOSEvalDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, ann_file, images_folder, overlap_window, max_clip_length, transforms, num_videos, force_bug=False):
        self.ann_file = ann_file
        self.num_videos = num_videos
        self.annotations = self._load_annotations()
        self.max_clip_length = max_clip_length
        self.overlap_window = overlap_window

        self.has_gt = "annotations" in self.annotations and self.annotations["annotations"] is not None


        self.cat_names = {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
        self.cat_names[0] = "Bkg"

        self._data = self.parse_video_into_clips(transforms, images_folder,force_bug)

    def _load_annotations(self):
        with open(self.ann_file, 'r') as fh:
            annotations = json.load(fh)
        has_gt = "annotations" in annotations
        if has_gt:
            has_gt = annotations["annotations"] is not None
        if has_gt and self.num_videos is not None and len(annotations["videos"]) > self.num_videos:
            annotations["videos"] = annotations["videos"][:self.num_videos]
            videos_ids = [video["id"] for video in annotations["videos"]]
            annotations["annotations"] = [annot for annot in annotations["annotations"] if annot["video_id"] in videos_ids]

        return annotations


    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def parse_video_into_clips(self, transforms, images_folder, force_bug):
        parsed_videos = []
        videos = self.annotations["videos"]

        for i in range(len(videos)):

            id_ = videos[i]['id']
            video_length = videos[i]['length']
            file_names = videos[i]['file_names']

            video_clips = []
            last_real_idx = 0
            real_video_length = None
            #TODO: Check this
            final_video_length = len(file_names)

            if video_length < self.max_clip_length:
                if force_bug:
                    video_to_read = []
                    i = 1
                    video_to_read.extend(file_names)
                    while len(video_to_read) < self.max_clip_length:
                        if i % 2:
                            video_to_read.extend(file_names[::-1][1:])
                        else:
                            video_to_read.extend(file_names[1:])
                        i += 1
                    video_clips.append(video_to_read[:self.max_clip_length])
                    real_video_length = video_length

                else:
                    video_to_read = []
                    j = 1
                    video_to_read.extend(file_names)
                    while len(video_to_read) < self.max_clip_length:
                        if j % 2:
                            video_to_read.extend(file_names[::-1][1:])
                        else:
                            video_to_read.extend(file_names[1:])
                        j += 1
                    video_clips.append(video_to_read[:self.max_clip_length])
                    real_video_length = video_length

            elif video_length == self.max_clip_length:
                clip_names = file_names[:self.max_clip_length]
                video_clips.append(clip_names)

            else:
                first_clip = file_names[:self.max_clip_length]
                video_clips.append(first_clip)

                next_start_pos = self.max_clip_length - self.overlap_window
                next_end_pos = next_start_pos + self.max_clip_length

                while next_end_pos < video_length:
                    next_video_clip = file_names[next_start_pos:next_end_pos]
                    video_clips.append(next_video_clip)
                    next_start_pos = next_end_pos - self.overlap_window
                    next_end_pos = next_start_pos + self.max_clip_length

                last_clip_start_idx = len(file_names) - 1 - self.max_clip_length
                last_real_idx = next_start_pos - last_clip_start_idx - 1
                last_video_clip = file_names[-self.max_clip_length:]
                video_clips.append(last_video_clip)

            original_size = (videos[i]['height'], videos[i]['width'])
            parsed_videos.append(YTVOSVideoClip(video_id=id_, video_clips=video_clips, last_real_idx=last_real_idx, original_size=original_size, real_video_length=real_video_length,
                           transform=transforms, images_folder=images_folder, final_video_length=final_video_length, cat_names=self.cat_names))

        return parsed_videos


def pil_image_load(image_path):
    return Image.open(image_path).convert('RGB')

def opencv_image_load(image_path):
    return cv2.imread(image_path)


def get_transform_pipeline(image_set, transform_pipeline, transform_strategy, max_size, out_scale, val_width, use_non_valid_class, create_bbx_from_mask, use_instance_level_classes):

    if image_set == "train":
        if transform_pipeline == "optimized_transform":
            print("Using optimized transforms")
            transforms = ClipTransformsApplier(transform_strategy, max_size, out_scale, use_non_valid_class, create_bbx_from_mask, use_instance_level_classes)
            prepare = ConvertCocoPolysToValuedMaskNumpy()
            load_func = opencv_image_load

        elif transform_pipeline == "default":
            print("Using default transforms")
            prepare = ConvertCocoPolysToMask(True)
            load_func = pil_image_load
            transforms = make_coco_transforms("train")

        else:
            raise ValueError("Select one transformation pipeline")

        return load_func, prepare, transforms

    else:
        if transform_strategy == "defdetr":
            defdetr_max_size = 1333
            defdetr_val_width = 800
            scale = val_width / defdetr_val_width
            max_size = int(scale * defdetr_max_size)

        elif transform_strategy == "defdetr_lower_res":
            val_width = 360
            max_size = 640

        elif transform_strategy == "ovis":
            val_width = 360
            max_size = 640

        elif transform_strategy == "seqformer":
            val_width = 360
            max_size = 640

        elif transform_strategy == "vistr":
            out_shorter_edge = ([int(300 * out_scale)], int(540 * out_scale))
            val_width = out_shorter_edge[0]
            max_size = out_shorter_edge[1]

        transform = torch_T.Compose([
                T.RandomResize([val_width], max_size=max_size),
                torch_T.ToTensor(),
                torch_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        return transform



def build(image_set, args, split, num_videos_to_eval):

    if split is None:
        split = getattr(args, f"{image_set}_set")

    root = Path(args.data_path)
    assert root.exists(), f'provided Data path {root} does not exist'

    PATHS = {
        "yt_vis_train": ((root / "Youtube_VIS/train/JPEGImages", root / "Youtube_VIS/train/" / 'train.json'), 40),
        "ovis_train": ((root / "OVIS/train/", root / "OVIS/" / "annotations_train.json"), 25),
        "ovis_val": ((root / "OVIS/valid/", root / "OVIS/" / "annotations_valid.json"), 25),
        "yt_vis_val_19": ((root / "Youtube-VOS/valid/JPEGImages", root / "Youtube_VIS/valid/" / "valid.json"), 40),
        "train_train_val_split": ((root / "Youtube_VIS/train/JPEGImages", root / "Youtube_VIS/train/" / 'train_train_val_split.json'), 40),
        "valid_train_val_split": ((root / "Youtube_VIS/train/JPEGImages", root / "Youtube_VIS/train/" / 'valid_train_val_split.json'), 40),
        "mini_train": ((root / "Youtube_VIS/train/JPEGImages", root / "Youtube_VIS/train/" / 'mini_train.json'), 40),
        "mini_val": ((root / "Youtube_VIS/valid/JPEGImages", root / "Youtube_VIS/valid/" / 'mini_valid.json'), 40),
        "yt_vis_train_2021": ((root / "Youtube_VIS-2021/train/JPEGImages", root / "Youtube_VIS-2021/train/" / 'instances.json'), 40),
        "yt_vis_val_2021": ((root / "Youtube_VIS-2021/valid/JPEGImages", root / "Youtube_VIS-2021/valid/" / 'instances.json'), 40),

    }
    img_folder, ann_file = PATHS[split][0]
    num_classes = PATHS[split][1]
    if image_set == "train":
        load_func, prepare, transforms = get_transform_pipeline("train", args.transform_pipeline,  args.transform_strategy, args.max_size, args.out_scale, None, args.use_non_valid_class, args.create_bbx_from_mask, args.use_instance_level_classes)
        dataset = YTVOSDataset(img_folder, ann_file, transforms=transforms, load_image_func=load_func, prepare_targets=prepare,
                               return_masks=args.masks, num_frames=args.num_frames, temporal_coherence=args.temporal_coherence,
                               reversed_sampling=args.reversed_sampling, focal_loss=args.focal_loss, use_non_valid_class=args.use_non_valid_class)
    else:
        transform = get_transform_pipeline("val", None,  args.transform_strategy, args.max_size, args.out_scale, args.val_width, False, False, False)
        dataset = YTVOSEvalDataset(ann_file, img_folder, args.overlap_window,  args.num_frames, transform, num_videos_to_eval, args.force_bug)

    return dataset, num_classes