import torch.utils.data
import torchvision

from .coco import build as build_coco
from .ytvos import build as build_vis
from pathlib import Path
import json


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, split=None, num_videos_to_eval=None):
    if args.dataset_type == 'coco':
        return build_coco(image_set, args)

    if args.dataset_type == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)

    if args.dataset_type == 'vis':
        return build_vis(image_set, args, split, num_videos_to_eval)

    raise ValueError(f'dataset type {args.dataset_file} not supported')


def read_test_set(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Data path {root} does not exist'
    mode = 'instances'

    PATHS = {
        "yt_vis_val": ((root / "Youtube-VOS/valid/JPEGImages", root / "Youtube_VIS/valid/" / f'{mode}.json'), 40),
        "ovis_val": ((root / "OVIS/valid/", root / "OVIS/" / "annotations_valid.json"), 25),
        "yt_vis_val_19": ((root / "Youtube-VOS/valid/JPEGImages", root / "Youtube_VIS/valid/" / "valid.json"), 40),
    }

    img_folder, ann_file = PATHS[image_set][0]
    val_dataset = json.load(open(ann_file, 'rb'))
    val_videos = val_dataset['videos']
    cat_names = {cat["id"]: cat["name"] for cat in val_dataset["categories"]}
    cat_names[0] = "Bkg"
    return val_videos, img_folder, PATHS[image_set][1], cat_names
