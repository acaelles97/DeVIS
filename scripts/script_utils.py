"""
YoutubeVIS data loader
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import sys
sys.path.append("/usr/prakt/p028/projects/VisTR/cocoapi/PythonAPI")
# from pycocotools_.ytvos import YTVOS
# from pycocotools_.ytvoseval import YTVOSeval
import datasets.transforms as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import random
import numpy as np




def convert_coco_poly_to_bool_mask(segmentation, height, width):
    if segmentation is None:
        return torch.zeros((1, height, width), dtype=torch.bool)

    if isinstance(segmentation['counts'], list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        mask = coco_mask.decode(rles)

    elif isinstance(segmentation['counts'], str):
        mask = coco_mask.decode(segmentation)

    else:
        raise ValueError("Error reading mask format")

    if len(mask.shape) < 3:
        mask = torch.as_tensor(mask[None, ...], dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool)

    return mask

def convert_coco_poly_to_bool_mask_numpy(segmentation, height, width):
    if segmentation is None:
        return np.zeros((height, width, 1), dtype=np.bool_)

    if isinstance(segmentation['counts'], list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        mask = coco_mask.decode(rles)

    elif isinstance(segmentation['counts'], str):
        mask = coco_mask.decode(segmentation)

    else:
        raise ValueError("Error reading mask format")

    if len(mask.shape) < 3:
        mask = mask[..., None].astype(dtype=np.bool_)
    else:
        mask = mask.astype(dtype=np.bool_)

    return mask



class ConvertCocoPolysToValuedMask(object):

    def __call__(self, image, target, inds, num_frames):
        w, h = image.size
        image_id = target["image_id"]
        frame_id = target['frame_id']
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = torch.zeros((num_frames*len(anno), 4), dtype=torch.float32)
        classes = torch.zeros((num_frames*len(anno)), dtype=torch.int64)
        segmentations = torch.zeros((num_frames, 1, h, w), dtype=torch.uint8)
        area = torch.zeros((num_frames*len(anno)))
        iscrowd = torch.zeros((num_frames*len(anno)))
        valid = torch.zeros((num_frames*len(anno)))
        instances = []
        for j in range(num_frames):
            frame_instances = []
            for i, ann in enumerate(anno):

                current_idx = i * num_frames + j
                bbox = ann['bboxes'][frame_id-inds[j]]
                areas = ann['areas'][frame_id-inds[j]]
                segm = ann['segmentations'][frame_id-inds[j]]
                label = ann["category_id"]

                # for empty boxes
                if bbox is None or segm is None:
                    bbox = [0, 0, 0, 0]
                    areas = 0
                    valid[current_idx] = 0
                    label = 0

                else:
                    valid[current_idx] = 1
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes[current_idx] = torch.as_tensor(bbox, dtype=torch.float32)
                area[current_idx] = torch.as_tensor(areas)
                mask = convert_coco_poly_to_bool_mask(segm, h, w)

                segmentations[j, mask] = i+1
                classes[current_idx] = torch.as_tensor(label, dtype=torch.int64)
                iscrowd[current_idx] = torch.as_tensor(crowd)

        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        target = {"boxes": boxes, "labels": classes, "masks": segmentations, "image_id": image_id, "valid": valid, "area": area,
                  "iscrowd": iscrowd, "orig_size": torch.as_tensor([int(h), int(w)]), "size": torch.as_tensor([int(h), int(w)])}

        return target



class ConvertCocoPolysToValuedMaskNumpy(object):

    def __call__(self, image, target, inds, num_frames):
        h, w = image.shape[:2]
        image_id = target["image_id"]
        frame_id = target['frame_id']

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = np.zeros((num_frames, len(anno), 4), dtype=np.float32)
        classes = np.zeros((num_frames, len(anno)), dtype=np.int64)
        segmentations = np.zeros((num_frames,  h, w, 1), dtype=np.uint8)
        area = torch.zeros((num_frames, len(anno)))
        iscrowd = torch.zeros((num_frames, len(anno)))
        valid = torch.zeros((num_frames, len(anno)), dtype=torch.int64)
        tmp_identifier = []
        clip_instances = []
        for j in range(num_frames):
            frame_instances = []
            for i, ann in enumerate(anno):
                tmp_identifier.append(f"Instance {i} Frame {j}")
                # current_idx = i * num_frames + j
                bbox = ann['bboxes'][frame_id-inds[j]]
                areas = ann['areas'][frame_id-inds[j]]
                segm = ann['segmentations'][frame_id-inds[j]]

                label = ann["category_id"]

                # for empty boxes
                if bbox is None or segm is None:
                    bbox = [0, 0, 0, 0]
                    areas = 0
                    valid[j, i] = 0
                    label = 0

                else:
                    valid[j, i] = 1

                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes[j, i] = np.asarray(bbox, dtype=np.float32)
                area[j, i] = areas
                mask = convert_coco_poly_to_bool_mask_numpy(segm, h, w)
                if np.any(mask):
                    frame_instances.append(i + 1)
                segmentations[j, mask] = i + 1
                classes[j, i] = label
                iscrowd[j, i] = crowd
            clip_instances.append(frame_instances)
        boxes[:, :, 2:] += boxes[:, :, :2]
        boxes[:, :, 0::2] = boxes[:, :, 0::2].clip(min=0, max=w)
        boxes[:, :, 1::2] = boxes[:, :, 1::2].clip(min=0, max=h)

        target = {"boxes": boxes, "labels": classes, "masks": segmentations, "image_id": torch.tensor([image_id]), "valid": valid, "area": area,
                  "iscrowd": iscrowd, "orig_size": torch.as_tensor([int(h), int(w)]), "tmp_identifier": tmp_identifier, "clip_instances": clip_instances}

        return target