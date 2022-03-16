

import numpy as np
import os
import tqdm
import cv2
import torch
import pycocotools.mask as mask_tools

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
# from VisTR.util.box_ops import box_cxcywh_to_xyxy

def get_most_left_coordinate(mask):

    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    if horizontal_indicies.shape[0]:
        x1 = horizontal_indicies[0]
        y1 = np.where(mask[:, x1])[0]
        if y1.shape[0]:
            return y1[-1], x1

    return None



def un_normalize(tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor


def imshow_det_bboxes(img, targets, cmap,
                      min_th=0.6,
                      class_names=None,
                      thickness=2,
                      font_size=8,
                      win_name='COCO visualization',
                      out_file=None):

    text_color = (1, 1, 1)
    bbox_color = (1, 0, 0)


    img = un_normalize(img)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    EPS = 1e-2
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    polygons = []
    color = []

    # for idx, (mask, bbox, label) in enumerate(zip(targets[0], targets[1], targets[2])):
    #
    #     mask = mask.numpy()
    #     label = label.item()
    #     bbox_int = bbox.numpy().astype(np.int32)
    #     poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
    #             [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
    #     np_poly = np.array(poly).reshape((4, 2))
    #     polygons.append(Polygon(np_poly))
    #     color.append(bbox_color)
    #
    #     if mask is None:
    #         continue
    #     if not np.any(mask):
    #         continue
    #
    #     text_coords = get_most_left_coordinate(mask)
    #     label_text = class_names[label] if class_names is not None else f'class {label}'
    #     ax.text(text_coords[1],
    #         text_coords[0],
    #         f'{label_text}',
    #         bbox={
    #             'facecolor': 'black',
    #             'alpha': 0.8,
    #             'pad': 0.7,
    #             'edgecolor': 'none'
    #         },
    #         color=text_color,
    #         fontsize=font_size,
    #         verticalalignment='top',
    #         horizontalalignment='left')
    #     color_mask = cmap[idx]
    #     img[mask != 0] = img[mask != 0] * 0.3 + color_mask * 0.7

    plt.imshow(img)
    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')

    cv2.imwrite(out_file, img)

    plt.close()


def create_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def parse_targets_per_frame(num_annots, num_frames, targets):
    idx = []

    num_instances_per_frame = int(num_annots / num_frames)
    frame_ids = np.arange(0, num_annots, num_frames)
    for annot in range(num_frames):
        for frame_id in frame_ids:
            idx.append(int(frame_id + annot))

    # old_identifier = np.array(targets["tmp_identifier"])

    targets["masks"] = targets["masks"][idx]
    targets["boxes"] = targets["boxes"][idx]
    targets["labels"] = targets["labels"][idx]
    # targets["valid"] = np.array(targets["valid"])[idx]

    # new_identifier = list(old_identifier[idx])
    # old_identifier = list(old_identifier)

    return targets


def visualize_transformed_images(clip_images,  targets, out_path, class_name):

    clip_images = clip_images.split(3, dim=0)
    num_images = len(clip_images)
    targets = parse_targets_per_frame(targets["boxes"].shape[0], num_images, targets)
    # images = torch.chunk(clip_images, num_images, dim=0)
    targets["masks"] = torch.chunk(targets["masks"], num_images, dim=0)
    targets["boxes"] = torch.chunk(targets["boxes"], num_images, dim=0)
    targets["labels"] = torch.chunk(targets["labels"], num_images, dim=0)
    targets["valid"] = torch.chunk(targets["valid"], num_images, dim=0)

    cmap = create_color_map()

    os.makedirs(out_path, exist_ok=True)

    for idx, image in enumerate(clip_images):
        frame_results = (targets["masks"][idx], targets["boxes"][idx], targets["labels"][idx], targets["valid"][idx])

        out_image_path = os.path.join(out_path, f"img_{idx:04d}.jpg")
        imshow_det_bboxes(image, frame_results, cmap=cmap, class_names=class_name, out_file=out_image_path)