import albumentations.augmentations.functional as F
import random
import numpy as np
import cv2
from albumentations.augmentations.transforms import RandomBrightnessContrast, HueSaturationValue, ChannelShuffle
from albumentations.augmentations.transforms import HorizontalFlip, RandomCrop
from albumentations import Compose as AlbuCompose
from albumentations.pytorch.transforms import img_to_tensor, mask_to_tensor
from torchvision.transforms import Normalize, ToTensor
from torchvision.transforms import Compose as PyTorchCompose
from albumentations import BboxParams
from util.box_ops import box_xyxy_to_cxcywh
from numpy import random as rand

import torch


class CustomResize(object):

    def __call__(self, size, **kwargs):

        out = {"image": F.resize(kwargs["image"], size[0], size[1])}

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, kwargs["image"].shape))
        ratio_width, ratio_height = ratios
        out["bboxes"] = kwargs["bboxes"] * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height])
        # out["area"] = kwargs["area"] * (ratio_width * ratio_height)
        out["mask"] = F.resize(kwargs["mask"], size[0], size[1], interpolation=cv2.INTER_NEAREST)

        # if "bboxes" in kwargs:
        #     out["bboxes"] = kwargs["bboxes"] * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height])
        #
        if "area" in kwargs:
            out["area"] = kwargs["area"] * (ratio_width * ratio_height)
        #
        # if "mask" in kwargs:
        #     out["mask"] = F.resize(kwargs["mask"], size[0], size[1], interpolation=cv2.INTER_NEAREST_EXACT)

        out["size"] = np.asarray([size[0], size[1]])

        return out


# T.RandomHorizontalFlip(p=1),
#           T.RandomResize(scales, max_size=800),
#           T.PhotometricDistort(prob=1),
#           T.RandomResize([400, 500, 600]),
#           T.RandomSizeCrop(384, 600),
#           # To suit the GPU memory the scale might be different
#           T.RandomResize([300], max_size=540), #for r50
#           #T.RandomResize([280], max_size=504),#for r101
#           T.ToTensor(),
#           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),


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
            mask[uint_mask[:, :, 0] == (idx + 1)] = True
        masks.append(mask)

    return torch.stack(masks, dim=0)


def compute_region(in_size, min_size, max_size):
    h, w = in_size

    tw = random.randint(min_size, min(w, max_size))
    th = random.randint(min_size, min(h, max_size))

    if h + 1 < th or w + 1 < tw:
        raise ValueError(
            "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
        )

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1, )).item()
    j = torch.randint(0, w - tw + 1, size=(1, )).item()
    return i, j, th, tw


class PytorchToTensor:

    def __init__(self):
        self.image_transform = PyTorchCompose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, **kwargs):
        # target = {"boxes": boxes, "labels": classes, "masks": segmentations, "image_id": image_id, "valid": valid, "area": area,
        #           "iscrowd": iscrowd, "orig_size": torch.as_tensor([int(h), int(w)]), "size": torch.as_tensor([int(h), int(w)])}

        kwargs["image"] = self.image_transform(kwargs["image"])
        # TODO: FIX BBX TO CORRECT FORMAT
        h, w = kwargs["image"].shape[-2:]
        kwargs["boxes"] = torch.as_tensor(kwargs["bboxes"], dtype=torch.float32)
        kwargs["boxes"] = box_xyxy_to_cxcywh(kwargs["boxes"])
        kwargs["boxes"] = kwargs["boxes"] / torch.tensor([w, h, w, h], dtype=torch.float32)

        kwargs["labels"] = torch.as_tensor(kwargs["labels"], dtype=torch.int64)
        kwargs["masks"] = create_binary_masks(kwargs['boxes'].shape[0], kwargs["mask"])
        assert kwargs["masks"].shape[0] == kwargs["boxes"].shape[0]

        for i in range(kwargs["masks"].shape[0]):
            area = torch.sum(kwargs["masks"][i])
            if area == 0:
                kwargs["boxes"][i] = torch.zeros(4)
                kwargs["valid"][i] = torch.tensor(0)
                kwargs["labels"][i] = torch.tensor(0)
                kwargs["area"][i] = 0
            else:
                kwargs["area"][i] = area

        return kwargs

class PytorchToTensorNoBBXChange:

    def __init__(self):
        self.image_transform = PyTorchCompose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, **kwargs):
        # target = {"boxes": boxes, "labels": classes, "masks": segmentations, "image_id": image_id, "valid": valid, "area": area,
        #           "iscrowd": iscrowd, "orig_size": torch.as_tensor([int(h), int(w)]), "size": torch.as_tensor([int(h), int(w)])}

        kwargs["image"] = self.image_transform(kwargs["image"])
        # TODO: FIX BBX TO CORRECT FORMAT
        h, w = kwargs["image"].shape[-2:]
        kwargs["boxes"] = torch.as_tensor(kwargs["bboxes"], dtype=torch.float32)

        kwargs["labels"] = torch.as_tensor(kwargs["labels"], dtype=torch.int64)
        kwargs["masks"] = create_binary_masks(kwargs['boxes'].shape[0], kwargs["mask"])
        assert kwargs["masks"].shape[0] == kwargs["boxes"].shape[0]

        for i in range(kwargs["masks"].shape[0]):
            area = torch.sum(kwargs["masks"][i])
            if area == 0:
                kwargs["boxes"][i] = torch.zeros(4)
                kwargs["valid"][i] = torch.tensor(0)
                kwargs["labels"][i] = torch.tensor(0)
                kwargs["area"][i] = 0
            else:
                kwargs["area"][i] = area

        return kwargs

class ClipTransformsApplier:
    def __init__(self):
        self.scale_random_augm = ([480, 512, 544, 576, 608, 640, 672, 704, 736, 768], 800)
        self.safe_resize_augm = ([400, 500, 600], None)
        self.out_shorter_edge = ([300], 540)
        self.random_sized_crop = (384, 600)

        self.horizontal_flip = CustomHorizontalFlip()
        self.resize = CustomResize()
        self.photometric_transforms = AlbuPhotometricTransform(p=0.5)
        self.random_crop = CustomRandomCrop()
        self.to_tensor = PytorchToTensor()

    def merge_results(self, out_images, out_targets, out_frame):
        if out_images is None:
            out_images = out_frame["image"]
            out_targets = out_frame

        else:
            out_images = torch.cat([out_images, out_frame["image"]], dim=0)
            out_targets["masks"] = torch.cat([out_targets["masks"], out_frame["masks"]], dim=0)
            out_targets["boxes"] = torch.cat([out_targets["boxes"], out_frame["boxes"]], dim=0)
            out_targets["labels"] = torch.cat([out_targets["labels"], out_frame["labels"]], dim=0)
            out_targets["valid"] = torch.cat([out_targets["valid"], out_frame["valid"]], dim=0)
            out_targets["area"] = torch.cat([out_targets["area"], out_frame["area"]], dim=0)

        return out_images, out_targets

    def sort_per_instance(self, out_targets, num_frames):
        num_annots = out_targets["boxes"].shape[0]
        idx = []
        num_instances_per_frame = int(num_annots/num_frames)
        frame_ids = np.arange(0, num_annots, num_instances_per_frame)
        for i in range(num_instances_per_frame):
            for frame_id in frame_ids:
                idx.append(int(frame_id + i))

        out_targets["masks"] = out_targets["masks"][idx]
        out_targets["boxes"] = out_targets["boxes"][idx]
        out_targets["labels"] = out_targets["labels"][idx]
        out_targets["valid"] = out_targets["valid"][idx]
        out_targets["area"] = out_targets["area"][idx]

        # out_targets["valid"] = list(np.array(out_targets["valid"])[idx])

        # out_targets["tmp_identifier"] = list(np.array(out_targets["tmp_identifier"])[idx])

        del out_targets["image"]
        return out_targets

    def __call__(self, images, targets):
        image_res = images[0].shape
        num_frames = len(images)
        do_flip = random.random() < 0.5
        out_shape_random_augm_params = compute_resize_scales(self.scale_random_augm[0], image_res, max_size=self.scale_random_augm[1])
        out_shape_safe_resize_augm = compute_resize_scales(self.safe_resize_augm[0], out_shape_random_augm_params, max_size=self.safe_resize_augm[1])
        crop_region = compute_region(out_shape_safe_resize_augm, *self.random_sized_crop)
        output_shape_params = compute_resize_scales(self.out_shorter_edge[0], crop_region[2:], max_size=self.out_shorter_edge[1])

        out_targets = None
        out_images = None
        for i in range(len(images)):
            if targets["boxes"][i].shape[0] > 1:
                print("A")
            # Create a new
            out = self.horizontal_flip(image=images[i], bboxes=targets["boxes"][i], mask=targets["masks"][i])
            out.update(self.resize(out_shape_random_augm_params,  image=out["image"], bboxes=out["bboxes"], mask=out["mask"]))
            num_instances = targets["clip_instances"][i]
            # out = self.resize(out_shape_random_augm_params, image=images[i], bboxes=targets["boxes"][i], mask=targets["masks"][i])
            out.update(self.photometric_transforms(image=out["image"])["image"])
            out.update(self.resize(out_shape_safe_resize_augm, image=out["image"], bboxes=out["bboxes"], mask=out["mask"]))
            out.update(self.random_crop(crop_region, image=out["image"], bboxes=out["bboxes"], mask=out["mask"]))
            out.update(self.resize(output_shape_params, image=out["image"], bboxes=out["bboxes"], mask=out["mask"]))
            out.update(self.to_tensor(image=out["image"], bboxes=out["bboxes"], mask=out["mask"], labels=targets["labels"][i], valid=targets["valid"][i], area=targets["area"][i]))
            del out["mask"]
            del out["bboxes"]

        for i in range(len(images)):
            if targets["boxes"][i].shape[0] > 1:
                print("A")
            # Create a new
            out = {
                "image": images[i],
                "bboxes": targets["boxes"][i],
                "mask": targets["masks"][i],
                "labels": targets["labels"][i],
                "valid": targets["valid"][i],
                "area": targets["area"][i]
            }

            out.update(self.horizontal_flip(do_flip, **out))
            out.update(self.resize(out_shape_random_augm_params, **out))
            out.update(self.photometric_transforms(image=out["image"])["image"])
            out.update(self.resize(out_shape_safe_resize_augm, image=out["image"], bboxes=out["bboxes"], mask=out["mask"]))
            out.update(self.random_crop(crop_region, image=out["image"], bboxes=out["bboxes"], mask=out["mask"]))
            out.update(self.resize(output_shape_params, image=out["image"], bboxes=out["bboxes"], mask=out["mask"]))
            out.update(self.to_tensor(image=out["image"], bboxes=out["bboxes"], mask=out["mask"], labels=targets["labels"][i],
                                      valid=targets["valid"][i], area=targets["area"][i]))
            del out["mask"]
            del out["bboxes"]


            # out.update(self.to_tensor(image=out["image"], bboxes=out["bboxes"], mask=out["mask"], labels=targets["labels"][i]))
            out_images, out_targets = self.merge_results(out_images, out_targets, out)

        # out_targets["tmp_identifier"] = targets["tmp_identifier"]
        out_targets = self.sort_per_instance(out_targets, num_frames)
        out_targets["image_id"] = targets["image_id"]
        out_targets["iscrowd"] = targets["iscrowd"]
        out_targets["orig_size"] = targets["orig_size"]
        out_targets["size"] = torch.as_tensor(output_shape_params)

        return out_images, out_targets


class NoBBXFormatClipTransformer(ClipTransformsApplier):
    def __init__(self):
        super().__init__()
        self.to_tensor = PytorchToTensorNoBBXChange()

    def __call__(self, images, targets):
        num_frames = len(images)
        images, targets = super().__call__(images, targets)
        images = torch.stack(images.chunk(num_frames, dim=0), dim=0)
        return images, targets

class AlbuPhotometricTransform:
    def __init__(self, p):
        self.transform = AlbuCompose([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=(0.2, 0.5), p=p),
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=0, p=p),
            ChannelShuffle(p=0.3)
        ])

    def __call__(self, **kwargs):
        kwargs["image"] = self.transform(image=kwargs["image"])
        return kwargs


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
        return image, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta
        return image, target


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, target


class RandomHue(object):  #
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, clip, target):
        imgs = []
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, target = self.rand_brightness(img, target)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, target = distort(img, target)
            img, target = self.rand_light_noise(img, target)
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, target

class CustomRandomCrop:

    def __call__(self, region, **kwargs):

        i, j, h, w = region
        kwargs["image"] = kwargs["image"][i:i + h, j:j + w, ...]

        if "bboxes" in kwargs:
            boxes = kwargs["bboxes"]
            max_size = np.asarray([w, h], dtype=np.float32)
            cropped_boxes = boxes - np.asarray([j, i, j, i])
            cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)

            cropped_boxes = cropped_boxes.clip(min=0)
            area = np.prod((cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]), axis=1)
            kwargs["bboxes"] = cropped_boxes.reshape(-1, 4)
            kwargs["area"] = area

        if "mask" in kwargs:
            kwargs['mask'] = kwargs["mask"][i:i + h, j:j + w, ...]

        return kwargs


# class CustomHorizontalFlip:
#     def __call__(self, do_flip, image, target):
#
#         if do_flip:
#             kwargs["image"] = F.hflip(kwargs["image"])
#             h, w = kwargs["image"].shape[:2]
#
#             if "bboxes" in kwargs:
#                 boxes = kwargs["bboxes"]
#                 boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) +  np.array([w, 0, w, 0])
#                 kwargs["bboxes"] = boxes
#
#             if "mask" in kwargs:
#                 kwargs['mask'] = np.fliplr(kwargs['mask'])
#
#         return kwargs

class CustomHorizontalFlip:
    def __call__(self, do_flip, image, target):

        if do_flip:
            image = F.hflip(image)
            h, w = image.shape[:2]

            if "bboxes" in target:
                boxes = target["bboxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) +  np.array([w, 0, w, 0])
                target["bboxes"] = boxes

            if "mask" in target:
                target['mask'] = np.fliplr(target['mask'])

        return image, target

class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            # class_name = str(t.__class__).split(".")[-1].split(">")[0][:-1]
            # tic = time.perf_counter()
            image, target = t(image, target)
            # toc = time.perf_counter()

            # print(f"Class: {class_name}: {toc-tic}")
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
