import math
import numpy as np
import cv2
import torch
def compute_locations(h, w, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def centroid_distance(higher_res, bbox):
    centroid = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]) /  torch.tensor([higher_res[1], higher_res[0]])
    locations = compute_locations(higher_res[0], higher_res[1], stride=8)
    locations = (locations / locations.max(axis=0)[0]).reshape(1, higher_res[0], higher_res[1], 2)
    centroids_reshape = centroid.reshape(1, 1, 1, 2)
    relative_coords = centroids_reshape - locations
    relative_coords = torch.abs(relative_coords)
    return relative_coords

def centroid_distance_2(higher_res, bbox, stride=1):
    centroid = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    centroid = centroid * stride
    locations = compute_locations(higher_res[0], higher_res[1], stride=stride)
    locations = locations.reshape(1, higher_res[0], higher_res[1], 2)
    centroids_reshape = centroid.reshape(1, 1, 1, 2)
    relative_coords = centroids_reshape - locations
    relative_coords = torch.abs(relative_coords)
    relative_coords = relative_coords / torch.tensor([relative_coords[..., 0].max(), relative_coords[..., 1].max()])
    relative_coords = 1 - relative_coords

    return relative_coords


def centroid_distance_3(higher_res, bbox, stride=4):
    centroid = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    centroid = centroid * stride
    locations = compute_locations(higher_res[0], higher_res[1], stride=stride)
    locations = locations.reshape(1, higher_res[0], higher_res[1], 2)
    centroids_reshape = centroid.reshape(1, 1, 1, 2)
    relative_coords = centroids_reshape - locations
    relative_coords = torch.abs(relative_coords)
    relative_coords = relative_coords / torch.tensor([relative_coords[..., 0].max(), relative_coords[..., 1].max()])
    h, w = torch.tensor((bbox[3] - bbox[1]) / higher_res[1]), torch.tensor((bbox[2] - bbox[0]) / higher_res[0])
    torch.pow(torch.full_like(relative_coords[0, :, :, 0], h), relative_coords[0, :, :, 0])

    relative_coords = 1 - relative_coords
    print("A")

    return relative_coords


def centroid_distance_5(higher_res, bbox, stride=1):
    centroid = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

    shifts_x = torch.abs(centroid[0]- torch.arange(0, higher_res[1] * stride, step=stride, dtype=torch.float32))
    shifts_x = shifts_x / shifts_x.max()

    # grid = torch.stack((grid_x, grid_y), -1)

    mu = centroid[0] / higher_res[1]
    sigma = 0.2
    x = torch.arange(0, higher_res[1] * stride, step=stride, dtype=torch.float32) /  higher_res[1]
    values = []
    for value in x:
        y = 1 / ((sigma * math.sqrt(2*math.pi)) * torch.exp(-0.5 * torch.pow((value-mu)/sigma, 2)))
        values.append(y)

    return grid
def gaussian_radius(det_size, min_overlap=0.7):
  # min_overlap = 0.1
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def np_gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # y, x = np.arange(-m, m + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def np_draw_umich_gaussian(heatmap, center, radius, k=1):
    radius = int(radius * 1.2)
    diameter = (2 * radius + 1)
    gaussian = np_gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def np_prepare_heatmap(hm_h, hm_w, bbox):
    hm_disturb = 0.05
    down_ratio = 4 # output stride, only 4 supported
    lost_disturb = 0.4

    pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32)
    pre_cts, track_ids = [], []
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    max_rad = 1
    if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        max_rad = max(max_rad, radius)
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct0 = ct.copy()
        conf = 1
        ct_int = ct.astype(np.int32)
        pre_cts.append(ct0 / down_ratio)

        hp = np_draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

    return hp

# def torch_gaussian_radius(det_size, min_overlap=0.1):
#   # min_overlap = 0.1
#   height, width = det_size
#
#   a1  = 1
#   b1  = (height + width)
#   c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
#   sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
#   r1  = (b1 + sq1) / 2
#
#   a2  = 4
#   b2  = 2 * (height + width)
#   c2  = (1 - min_overlap) * width * height
#   sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
#   r2  = (b2 + sq2) / 2
#
#   a3  = 4 * min_overlap
#   b3  = -2 * min_overlap * (height + width)
#   c3  = (min_overlap - 1) * width * height
#   sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
#   r3  = (b3 + sq3) / 2
#   return min(r1, r2, r3)
#
# def torch_gaussian2D(shape, sigma=1):
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     y = torch.linspace(-m, m, int(m)*2+1, dtype=torch.float32, device=m.device)[:, None]
#     x = torch.linspace(-n, n, int(n)*2+1, dtype=torch.float32, device=n.device)[None]
#     h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
#     h[h < torch.finfo(h.dtype).eps * h.max()] = 0
#     return h
#
# def torch_draw_umich_gaussian(heatmap, center, radius, k=1):
#     diameter = (2 * radius + 1)
#     gaussian = torch_gaussian2D((diameter, diameter), sigma=diameter / 6)
#
#     x, y  = center[0], center[1]
#
#     height, width = heatmap.shape[0:2]
#
#     left, right = torch.min(x, radius), torch.min(width - x, radius + 1)
#     top, bottom = torch.min(y, radius), torch.min(height - y, radius + 1)
#     masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
#     masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
#     if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
#         torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
#     return heatmap
#
#
# def torch_prepare_heatmap(hm_h, hm_w, bbox):
#     pre_hm = torch.zeros((1, hm_h, hm_w), dtype=torch.float32)
#     pre_cts, track_ids = [], []
#     h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
#     max_rad = 1
#     if (h > 0 and w > 0):
#         radius = torch_gaussian_radius((h, w))
#         radius = torch.max(torch.tensor(0), radius.type(torch.int))
#         ct = torch.tensor(
#             [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=torch.float32)
#         conf = 1
#         ct = ct.type(torch.int)
#         hp = torch_draw_umich_gaussian(pre_hm[0], ct, radius, k=conf)
#
#     return hp

def torch_gaussian_radius(det_size, min_overlap=0.1):
  # min_overlap = 0.1
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return torch.min(torch.stack([r1, r2, r3], dim=1), dim=1)[0]

def torch_gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y = torch.linspace(-m, m, int(m)*2+1, dtype=torch.float32, device=m.device)[:, None]
    x = torch.linspace(-n, n, int(n)*2+1, dtype=torch.float32, device=n.device)[None]
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

def torch_draw_umich_gaussian(all_heatmap, all_center, all_radius, k=1):
    out_heatmap = []
    for i_bbox in range(all_center.shape[0]):
        radius = all_radius[i_bbox]
        center = all_center[i_bbox]
        heatmap = all_heatmap[i_bbox]

        diameter = (2 * radius + 1)
        gaussian = torch_gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y  = center[0], center[1]

        height, width = heatmap.shape[0:2]

        left, right = torch.min(x, radius), torch.min(width - x, radius + 1)
        top, bottom = torch.min(y, radius), torch.min(height - y, radius + 1)
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        out_heatmap.append(heatmap)

    return torch.stack(out_heatmap, dim=0)


def torch_prepare_heatmap(hm_h, hm_w, boxes):
    num_boxes = boxes.shape[0]
    pre_hm = torch.zeros((num_boxes, hm_h, hm_w), dtype=torch.float32)
    h, w = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]
    h = torch.clamp(h, min=1, max=hm_h)
    w = torch.clamp(w, min=1, max=hm_w)
    radius = torch_gaussian_radius((h, w))
    radius = torch.max(torch.zeros_like(radius,dtype=torch.int), radius.type(torch.int))
    ct = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], dim=1)
    conf = 1
    ct = ct.type(torch.int)
    hp = torch_draw_umich_gaussian(pre_hm, ct, radius, k=conf)

    return hp


def save_image(hp, out_name, bbx):
    out_path = "/usr/stud/cad/results/inference/heatmap_exp/" + out_name
    hp = cv2.cvtColor(hp, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(hp, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (255, 0, 0), 1)
    hp = cv2.circle(hp, ((bbx[0] + bbx[2]) // 2, (bbx[1] + bbx[3]) // 2), 1, (0,0,255), 1)
    cv2.imwrite(out_path, hp)


if __name__ == "__main__":
    bboxes = torch.tensor([[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[50, 10, 80, 50], [12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20],[12, 10, 40, 20]])
    # mask = torch.zeros([72, 100], dtype=torch.bool)
    # mask[4:10, :] = True
    import time

    time1 = time.time()
    hp2_torch = torch_prepare_heatmap(72, 100, bboxes) * 255

    print(f"Time all {time.time()-time1}")


    time1 = time.time()
    for box in bboxes:
        hp2_torch = torch_prepare_heatmap(72, 100, box[None]) * 255

    print(f"Time for {time.time() - time1}")


    time1 = time.time()
    hp2_torch = torch_prepare_heatmap(72, 100, bboxes) * 255

    print(f"Time all {time.time()-time1}")


    time1 = time.time()
    for box in bboxes:
        hp2_torch = torch_prepare_heatmap(72, 100, box[None]) * 255

    print(f"Time for {time.time() - time1}")

    time1 = time.time()
    hp2_torch = torch_prepare_heatmap(72, 100, bboxes) * 255

    print(f"Time all {time.time() - time1}")

    time1 = time.time()
    for box in bboxes:
        hp2_torch = torch_prepare_heatmap(72, 100, box[None]) * 255

    print(f"Time for {time.time() - time1}")

    time1 = time.time()
    hp2_torch = torch_prepare_heatmap(72, 100, bboxes) * 255

    print(f"Time all {time.time() - time1}")

    time1 = time.time()
    for box in bboxes:
        hp2_torch = torch_prepare_heatmap(72, 100, box[None]) * 255

    print(f"Time for {time.time() - time1}")

    time1 = time.time()
    hp2_torch = torch_prepare_heatmap(72, 100, bboxes) * 255

    print(f"Time all {time.time() - time1}")

    time1 = time.time()
    for box in bboxes:
        hp2_torch = torch_prepare_heatmap(72, 100, box[None]) * 255

    print(f"Time for {time.time() - time1}")

    for i, box in enumerate(bboxes):
        bbx_hp = (np_prepare_heatmap(72, 100,  box) * 255 ).astype(np.uint8)
        # save_image(bbx_hp, str(i)+"_old_hp.jpg", box)

        old_center_distance = centroid_distance([72, 100], box) * 255
        # save_image(old_center_distance[0,:,:,0].numpy().astype(np.uint8), str(i)+"_old_ct_x.jpg", box)
        # save_image(old_center_distance[0,:,:,1].numpy().astype(np.uint8), str(i)+"_old_ct_y.jpg", box)

        new_center_distance = centroid_distance_2([72, 100], box)
        new_center_distance = torch.pow((new_center_distance[0,:,:,1] * new_center_distance[0,:,:,0]), 4)

        new_center_distance.masked_fill_(mask, 0)
        new_center_distance = new_center_distance * 255

        save_image(new_center_distance.numpy().astype(np.uint8), str(i)+"_new_ct.jpg", box)

        # save_image(new_center_distance[0,:,:,0].numpy().astype(np.uint8), str(i)+"_new_ct_x.jpg", box)
        # save_image(new_center_distance[0,:,:,1].numpy().astype(np.uint8), str(i)+"_new_ct_y.jpg", box)


    # distance_hp_x =
    # distance_hp_y = distance_hp[0,:,:,1].numpy().astype(np.uint8)
    #
    #
    #
    #
    # hp2_torch = cv2.cvtColor(distance_hp_x, cv2.COLOR_GRAY2RGB)
    # cv2.rectangle(hp2_torch, (50, 10), (80, 50), (255, 0, 0), 1)
    # hp2_torch = cv2.circle(hp2_torch, ((50 + 80) // 2, (10+50)// 2), 1, (0,0, 255), 1)
    # cv2.imwrite('/usr/stud/cad/results/inference/heatmap_exp/distance_x.jpg', hp2_torch)
    #
    # hp2_torch_y = cv2.cvtColor(distance_hp_y.numpy().astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # cv2.rectangle(hp2_torch_y, (50, 10), (80, 50), (255, 0, 0), 1)
    # hp2_torch_y = cv2.circle(hp2_torch_y, ((50 + 80) // 2, (10+50)// 2), 1, (0,0, 255), 1)
    # cv2.imwrite('/usr/stud/cad/results/inference/heatmap_exp/distance_y.jpg', hp2_torch_y)
    #
    #
    # # hp2_torch = (torch_prepare_heatmap(72, 100,  torch.tensor([50, 10, 80, 50])) * 255 )
    # # # hp3_torch = (torch_prepare_heatmap(72, 100,  torch.tensor([50, 10, 80, 50])) * 255 )
    # #
    # # # same1 = np.all(hp2_torch.numpy().astype(np.uint8) == hp2)
    # # same2 = np.all(hp2_torch.numpy().astype(np.uint8) == hp3)
    # #
    # # #
    #
    #
    # #
    # # hp2_torch = cv2.cvtColor(hp2_torch.numpy().astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # # cv2.rectangle(hp2_torch, (50, 10), (80, 50), (255, 0, 0), 1)
    # # hp2_torch = cv2.circle(hp2_torch, ((50 + 80) // 2, (10+50)// 2), 1, (0,0, 255), 1)
    #
    # #
    # cv2.imwrite('/usr/stud/cad/results/inference/heatmap_exp/heatmap.jpg', hp3)
    # cv2.imwrite('/usr/stud/cad/results/inference/heatmap_exp/distance_x.jpg', hp2_torch)
