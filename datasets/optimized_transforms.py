import random
import numpy as np
import cv2
from torchvision.transforms import Normalize, ToTensor
from torchvision.transforms import Compose as PyTorchCompose
from datasets.transforms import RandomContrast, ConvertColor, RandomSaturation, RandomHue, RandomBrightness, RandomLightingNoise, Compose
from util.box_ops import box_xyxy_to_cxcywh, masks_to_boxes
import warnings
from pycocotools import mask as coco_mask

import torch



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



def compute_resize_params(image_size, size, max_size):
    h, w = image_size[:2]
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return oh, ow


def compute_resize_scales(scales, image_size, max_size):
    size = random.choice(scales)
    if isinstance(size, (list, tuple)):
        return size[::-1]
    else:
        return compute_resize_params(image_size, size, max_size)


def create_binary_masks(num_instances, uint_mask):
    masks = []
    height, width = uint_mask.shape[:2]
    unique_instances = set(np.unique(uint_mask)) - {0}
    for idx in range(num_instances):
        mask = torch.zeros(height, width, dtype=torch.bool)
        if (idx+1) in unique_instances:
            if len(uint_mask.shape) == 3:
                mask[uint_mask[:, :, 0] == (idx + 1)] = True
            else:
                mask[uint_mask == (idx + 1)] = True

        masks.append(mask)

    return torch.stack(masks, dim=0)


def compute_region(in_size, min_size, max_size):
    h, w = in_size
    if min_size > min(w, max_size) or min_size > min(h, max_size):
        return -1, -1, h, w

    tw = random.randint(min_size, min(w, max_size))
    th = random.randint(min_size, min(h, max_size))

    if h + 1 < th or w + 1 < tw:
        warnings.warn(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")
        return -1, -1, h, w

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1, )).item()
    j = torch.randint(0, w - tw + 1, size=(1, )).item()
    return i, j, th, tw


class ToTensorWithProcessing:

    def __init__(self, create_bbx_from_mask):
        self.image_transform = PyTorchCompose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.create_bbx_from_mask = create_bbx_from_mask

    def __call__(self, image, target):
        image = self.image_transform(image)
        # TODO: FIX BBX TO CORRECT FORMAT
        h, w = image.shape[-2:]
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["boxes"] = box_xyxy_to_cxcywh(target["boxes"])
        target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h], dtype=torch.float32)

        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        target["area"] = torch.as_tensor(target["area"], dtype=torch.int64)
        target["size"] = torch.as_tensor(target["size"], dtype=torch.int64)

        target["masks"] = create_binary_masks(target['boxes'].shape[0], target["masks"])
        num_objs = target["boxes"].shape[0]
        assert target["masks"].shape[0] == target["boxes"].shape[0]
        target["centroids"] = torch.zeros(num_objs, 2)
        for i in range(target["masks"].shape[0]):
            area = torch.sum(target["masks"][i])
            if area <= 5:
                target["boxes"][i] = torch.zeros(4)
                target["valid"][i] = torch.tensor(0)
                target["labels"][i] = torch.tensor(0)
                target["area"][i] = torch.tensor(0)
            else:
                if self.create_bbx_from_mask:
                    new_bbx = masks_to_boxes(target["masks"][i][None])
                    new_bbx = box_xyxy_to_cxcywh(new_bbx)
                    target["boxes"][i] = new_bbx / torch.tensor([w, h, w, h], dtype=torch.float32)

                centroid_yx = torch.mean(target["masks"][i].nonzero().type(torch.float32), dim=0).type(torch.int)
                centroid = torch.tensor([centroid_yx[1], centroid_yx[0]])
                target["centroids"][i] = centroid / torch.tensor([w, h])
                target["area"][i] = area

        return image, target


class CustomResize(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.out_size = None

    def init_clip_transform(self, **kwargs):
        size = kwargs["size"]
        self.out_size = compute_resize_scales(self.sizes[0], size, max_size=self.sizes[1])
        kwargs["size"] = self.out_size
        if "previousReshape_1" not in kwargs:
            kwargs["previousReshape_1"] = size
        else:
            kwargs["previousReshape_2"] = size
        return kwargs

    def __call__(self, image, target):
        # image = F.resize(image, self.out_size[0], self.out_size[1])
        original_shape = image.shape[:2]
        image = cv2.resize(image, (self.out_size[1], self.out_size[0]), interpolation=cv2.INTER_LINEAR)

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(self.out_size, original_shape))
        ratio_height, ratio_width = ratios

        if "boxes" in target:
            target["boxes"] = target["boxes"] * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height])

        if "masks" in target:
            # target["masks"] = F.resize(target["masks"], self.out_size[0], self.out_size[1], interpolation=cv2.INTER_NEAREST)
            target["masks"] = cv2.resize(target["masks"], (self.out_size[1], self.out_size[0]), interpolation=cv2.INTER_NEAREST)

        if "area" in target:
            target["area"] = target["area"] * (ratio_width * ratio_height)

        target["size"] = np.asarray([self.out_size[0], self.out_size[1]])

        return image, target


class CustomRandomCrop:
    def __init__(self, size):
        self.size = size
        self.region = None

    def init_clip_transform(self, **kwargs):
        size = kwargs["size"]
        self.region = compute_region(size, *self.size)
        kwargs["size"] = self.region[2:]
        kwargs["previousShape_crop"] = size
        return kwargs

    def __call__(self, image, target):
        i, j, h, w = self.region
        # Check case in which image is shorter than the crop, just return then
        if i == -1 or j == -1:
            return  image, target

        image = image[i:i + h, j:j + w, ...]

        if "boxes" in target:
            boxes = target["boxes"]
            max_size = np.asarray([w, h], dtype=np.float32)
            cropped_boxes = boxes - np.asarray([j, i, j, i])
            cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)

            cropped_boxes = cropped_boxes.clip(min=0)
            area = np.prod((cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]), axis=1)
            target["boxes"] = cropped_boxes.reshape(-1, 4)
            target["area"] = area

        if "masks" in target:
            target['masks'] = target["masks"][i:i + h, j:j + w, ...]

        return image, target


class CustomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        self.do_flip = None

    def init_clip_transform(self, **kwargs):
        self.do_flip = random.random() < self.p
        kwargs["do_flip"] = self.do_flip
        return kwargs

    def __call__(self, image, target):

        if self.do_flip:
            image = np.ascontiguousarray(image[:, ::-1, ...])
            h, w = image.shape[:2]

            if "boxes" in target:
                boxes = target["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) + np.array([w, 0, w, 0])
                target["boxes"] = boxes

            if "masks" in target:
                target['masks'] = np.fliplr(target['masks'])

        return image, target


class PhotometricDistort(object):
    def __init__(self, p=0.5):
        self.pd = [
            RandomContrast(upper=1.3),
            ConvertColor(transform='HSV'),
            RandomSaturation(lower=0.7, upper=1.3),
            RandomHue(delta=8.0),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness(delta=10)
        self.rand_light_noise = RandomLightingNoise()
        self.p = p

    def __call__(self, image, target):
        image = image.astype('float32')
        # image, target = self.rand_brightness(image, target)
        if random.random() < self.p:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])

        image, target = distort(image, target)
        if random.random() < self.p:
            image, target = self.rand_light_noise(image, target)

        image = image.astype('uint8')
        return image, target


class CustomRandomSelect:
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transform1 = transforms1
        self.transform2 = transforms2
        self.p = p
        self.do_transforms1 = None

    def init_clip_transform(self, **transforms_kwargs):
        self.do_transforms1 = random.random() < self.p
        if self.do_transforms1:
            if hasattr(self.transform1, "init_clip_transform"):
                transforms_kwargs.update(self.transform1.init_clip_transform(**transforms_kwargs))
        else:
            if hasattr(self.transform2, "init_clip_transform"):
                transforms_kwargs.update(self.transform2.init_clip_transform(**transforms_kwargs))

        return transforms_kwargs

    def __call__(self, image, target):
        if self.do_transforms1:
            image, target = self.transform1(image, target)
        else:
            image, target = self.transform2(image, target)

        return image, target

class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def init_clip_transform(self, **transforms_kwargs):
        for t in self.transforms:
            if hasattr(t, "init_clip_transform"):
                transforms_kwargs.update(t.init_clip_transform(**transforms_kwargs))

        return transforms_kwargs

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target

class ClipTransformsApplier:
    def __init__(self, transform_strategy, max_size, out_scale, use_non_valid_class, create_bbx_from_mask, use_instance_level_classes):
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
        scales_before_crop = [400, 500, 600]
        random_sized_crop = (384, 600)
        self.use_non_valid_class = use_non_valid_class
        self.use_instance_level_classes = use_instance_level_classes
        self.target_keys = ["masks", "boxes", "labels", "valid", "area"]

        if transform_strategy == "vistr":
            if out_scale != 1.0:
                # scale all with respect to custom max_size
                scales = [int(out_scale * s) for s in scales]
                scales_before_crop = [int(out_scale * s) for s in scales_before_crop]
                random_sized_crop = tuple([int(out_scale * s) for s in random_sized_crop])
                max_size = int(max_size * out_scale)

            out_shorter_edge = ([int(300 * out_scale)], int(540 * out_scale))

            scales = (scales, max_size)
            scales_before_crop = (scales_before_crop, None)

            self.transforms = [
                CustomHorizontalFlip(),
                CustomResize(scales),
                PhotometricDistort(),
                CustomResize(scales_before_crop),
                CustomRandomCrop(random_sized_crop),
                CustomResize(out_shorter_edge),
                ToTensorWithProcessing(create_bbx_from_mask),
            ]

        elif transform_strategy == "defdetr":
            defdetr_max_size = 1333
            if max_size != defdetr_max_size:
                scale = max_size / defdetr_max_size
                # scale all with respect to custom max_size
                scales = [int(scale * s) for s in scales]
                scales_before_crop = [int(scale * s) for s in scales_before_crop]
                random_sized_crop = [int(scale * s) for s in random_sized_crop]

            scales = (scales, max_size)
            scales_before_crop = (scales_before_crop, None)

            self.transforms = [
                CustomHorizontalFlip(),
                # TODO: Check impact of photometric on this scenario
                # PhotometricDistort(),
                CustomRandomSelect(
                    CustomResize(scales),
                    CustomCompose([
                        CustomResize(scales_before_crop),
                        CustomRandomCrop(random_sized_crop),
                        CustomResize(scales),
                    ])
                ),
                ToTensorWithProcessing(create_bbx_from_mask),
            ]

        elif transform_strategy == "seqformer":
            scales = [288, 320, 352, 392, 416, 448, 480, 512]
            scales = (scales, 768)
            scales_before_crop = (scales_before_crop, None)

            self.transforms = [
                CustomHorizontalFlip(),
                PhotometricDistort(),
                CustomRandomSelect(
                    CustomResize(scales),
                    CustomCompose([
                        CustomResize(scales_before_crop),
                        CustomRandomCrop(random_sized_crop),
                        CustomResize(scales),
                    ])
                ),
                ToTensorWithProcessing(create_bbx_from_mask),
            ]

        elif transform_strategy == "ovis":
            scales = [288, 320, 352, 392, 416]
            scales = (scales, 678)
            scales_before_crop = (scales_before_crop, None)

            self.transforms = [
                CustomHorizontalFlip(),
                PhotometricDistort(),
                CustomRandomSelect(
                    CustomResize(scales),
                    CustomCompose([
                        CustomResize(scales_before_crop),
                        CustomRandomCrop(random_sized_crop),
                        CustomResize(scales),
                    ])
                ),
                ToTensorWithProcessing(create_bbx_from_mask),
            ]


        elif transform_strategy == "defdetr_lower_res":
            scales = [480, 512, 544, 576, 608, 640, 672, 704]
            max_size = 768
            defdetr_max_size = 1333
            if max_size != defdetr_max_size:
                scale = 800 / defdetr_max_size
                # scale all with respect to custom max_size
                scales = [int(scale * s) for s in scales]
                scales_before_crop = [int(scale * s) for s in scales_before_crop]
                random_sized_crop = [int(scale * s) for s in random_sized_crop]

            scales = (scales, max_size)
            scales_before_crop = (scales_before_crop, None)

            self.transforms = [
                CustomHorizontalFlip(),
                # TODO: Check impact of photometric on this scenario
                PhotometricDistort(),
                CustomRandomSelect(
                    CustomResize(scales),
                    CustomCompose([
                        CustomResize(scales_before_crop),
                        CustomRandomCrop(random_sized_crop),
                        CustomResize(scales),
                    ])
                ),
                ToTensorWithProcessing(create_bbx_from_mask),
            ]

        elif transform_strategy == "defdetr_lower_res_photometric":
            scales = [480, 512, 544, 576, 608, 640, 672, 704]
            defdetr_max_size = 1333
            if max_size != defdetr_max_size:
                scale = max_size / defdetr_max_size
                # scale all with respect to custom max_size
                scales = [int(scale * s) for s in scales]
                scales_before_crop = [int(scale * s) for s in scales_before_crop]
                random_sized_crop = [int(scale * s) for s in random_sized_crop]

            scales = (scales, max_size)
            scales_before_crop = (scales_before_crop, None)

            self.transforms = [
                CustomHorizontalFlip(),
                # TODO: Check impact of photometric on this scenario
                PhotometricDistort(),
                CustomRandomSelect(
                    CustomResize(scales),
                    CustomCompose([
                        CustomResize(scales_before_crop),
                        CustomRandomCrop(random_sized_crop),
                        CustomResize(scales),
                    ])
                ),
                ToTensorWithProcessing(create_bbx_from_mask),
            ]

        else:
            raise NotImplementedError

    def sort_per_instance(self, out_targets, num_frames):
        target_keys = self.target_keys + ["centroids"]
        num_annots = out_targets["boxes"].shape[0]
        idx = []
        num_instances_per_frame = int(num_annots/num_frames)
        frame_ids = np.arange(0, num_annots, num_instances_per_frame)
        for i in range(num_instances_per_frame):
            for frame_id in frame_ids:
                idx.append(int(frame_id + i))

        for key in target_keys:
            out_targets[key] = out_targets[key][idx]

        final_indices = []
        for instance in range(num_instances_per_frame):
            if not torch.any(out_targets["valid"][instance*num_frames:(instance+1)*num_frames]):
                final_indices.append(torch.full((num_frames,), 0, dtype=torch.bool))
            else:
                final_indices.append(torch.full((num_frames,), 1, dtype=torch.bool))
        final_indices = torch.cat(final_indices)
        if not torch.all(final_indices):
            for key in target_keys:
                out_targets[key] = out_targets[key][final_indices]

        return out_targets

    def fill_box_non_valid_instances(self, targets, num_frames):
        num_annots = targets["boxes"].shape[0]
        num_instances_per_frame = int(num_annots/num_frames)
        for instance in range(num_instances_per_frame):
            trajectory_boxes = targets["boxes"][instance * num_frames: (instance + 1) * num_frames]
            valid_frames =  targets["valid"][instance * num_frames: (instance + 1) * num_frames]
            if not torch.all(valid_frames):
                new_instance_boxes = []
                for idx, valid in enumerate(valid_frames):
                    if not valid:
                        # Find closest valid box
                        non_zero_boxes = valid_frames.nonzero()
                        frame_pos = torch.argmin(torch.abs(idx - non_zero_boxes))
                        new_bbx = torch.clone(trajectory_boxes[non_zero_boxes[frame_pos]])
                        # Set width and height to 0
                        new_bbx[0, 2:] = 1e-6
                        new_instance_boxes.append(new_bbx)
                    else:
                        new_instance_boxes.append(trajectory_boxes[idx][None])

                targets["boxes"][instance * num_frames: (instance + 1) * num_frames] = torch.cat(new_instance_boxes)

        return targets

    def set_all_frames_valid(self, targets, num_frames):
        num_annots = targets["boxes"].shape[0]
        num_instances_per_frame = int(num_annots/num_frames)
        all_valid = torch.ones(num_frames, dtype=torch.bool)
        for instance in range(num_instances_per_frame):
            targets["valid"][instance * num_frames: (instance + 1) * num_frames] = all_valid
            trajectory_labels = targets["labels"][instance * num_frames: (instance + 1) * num_frames]
            label = trajectory_labels[trajectory_labels.nonzero()[0]].item()
            new_label = torch.full_like(trajectory_labels, label)
            targets["labels"][instance * num_frames: (instance + 1) * num_frames] = new_label
        return targets

    def set_instance_level_classes(self,  targets, num_frames):
        num_annots = targets["boxes"].shape[0]
        num_instances_per_frame = int(num_annots/num_frames)
        new_classes = []
        for instance in range(num_instances_per_frame):
            trajectory_labels = targets["labels"][instance * num_frames: (instance + 1) * num_frames]
            label = trajectory_labels[trajectory_labels.nonzero()[0]].item()
            new_classes.append(label)
        targets["labels"] = torch.tensor(new_classes, dtype=torch.int64)

        return targets

    def __call__(self, images, targets):
        transforms_kwargs = {
            "size": images[0].shape
        }
        for t in self.transforms:
            if hasattr(t, "init_clip_transform"):
                transforms_kwargs.update(t.init_clip_transform(**transforms_kwargs))
        num_frames = len(images)
        out_targets, out_images = [], []

        for i in range(len(images)):
            image = images[i]
            target = {key: targets[key][i] for key in self.target_keys}

            for t in self.transforms:
                image, target = t(image, target)

            out_images.append(image), out_targets.append(target)

        targets_keys = self.target_keys + ["centroids"]
        out_images = torch.cat(out_images, dim=0)
        out_targets = {key: torch.cat([target[key] for target in out_targets], dim=0) for key in targets_keys}
        out_targets = self.sort_per_instance(out_targets, num_frames)
        out_targets["original_valid"] = torch.clone(out_targets["valid"])

        if self.use_non_valid_class:
            out_targets = self.fill_box_non_valid_instances(out_targets, num_frames)
            out_targets = self.set_all_frames_valid(out_targets, num_frames)

        if self.use_instance_level_classes:
            out_targets = self.set_instance_level_classes(out_targets, num_frames)

        out_targets["image_id"] = targets["image_id"]
        out_targets["iscrowd"] = targets["iscrowd"]
        out_targets["orig_size"] = targets["orig_size"]
        out_targets["size"] = torch.as_tensor(transforms_kwargs["size"])

        return out_images, out_targets
