# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Submodule interface.
"""
import warnings
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CocoDetection

from .coco import build as build_coco
from .vis import build as build_vis


def get_coco_api_from_dataset(dataset: Subset) -> COCO:
    """Return COCO class from PyTorch dataset for evaluation with COCO eval."""
    for _ in range(10):
        # if isinstance(dataset, CocoDetection):
        #     break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, CocoDetection):
        raise NotImplementedError

    return dataset.coco


def build_dataset(image_set, cfg):
    if cfg.DATASETS.TYPE == 'coco':
        return build_coco(image_set, cfg)

    if cfg.DATASETS.TYPE == 'coco_panoptic':
        warnings.warn("COCO panoptic has not been tested on this implementation", UserWarning)
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, cfg)

    if cfg.DATASETS.TYPE == 'vis':
        return build_vis(image_set, cfg)

    raise ValueError(f'dataset type {cfg.DATASETS.TYPE} not supported')
