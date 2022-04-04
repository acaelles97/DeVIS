# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Benchmark inference speed of Deformable DETR.
"""
import os
import time
import argparse

import torch

from main import get_args_parser as get_main_args_parser
from models import build_model
from datasets import build_dataset
from util.misc import nested_tensor_from_tensor_list, val_collate
from torch.utils.data import DataLoader, DistributedSampler


def get_benckmark_arg_parser():
    parser = argparse.ArgumentParser('Benchmark inference speed of Deformable DETR.')
    parser.add_argument('--num_iters', type=int, default=300, help='total iters to benchmark speed')
    parser.add_argument('--warm_iters', type=int, default=5, help='ignore first several iters that are very slow')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in inference')
    parser.add_argument('--resume', type=str, help='load the pre-trained checkpoint')
    return parser


@torch.no_grad()
def measure_average_inference_time(model, data_loader_val, num_frames, num_iters=100, warm_iters=5):
    ts = []
    iter = 0
    for idx, video in  enumerate(data_loader_val):
        if iter > num_iters:
            break
        sampler_val = torch.utils.data.SequentialSampler(video)
        video_loader = DataLoader(video, 1, sampler=sampler_val, num_workers=0)
        real_video_length = video.real_video_length
        clip_length = num_frames if real_video_length is None or real_video_length >= num_frames else real_video_length
        video_info = {
            "tgt_size": video.original_size,
            "clip_length": clip_length,
            "process_boxes": False,
        }
        for video_clip in video_loader:
            video_clip = video_clip.cuda().squeeze(0)
            torch.cuda.synchronize()
            t_ = time.perf_counter()
            model(video_clip, video_info)
            torch.cuda.synchronize()
            t = time.perf_counter() - t_
            iter += 1
            if iter >= warm_iters:
              ts.append(t / num_frames)

    print(ts)
    return sum(ts) / len(ts)


def benchmark():
    args, _ = get_benckmark_arg_parser().parse_known_args()
    main_args = get_main_args_parser().parse_args(_)
    assert args.warm_iters < args.num_iters and args.num_iters > 0 and args.warm_iters >= 0
    assert args.batch_size > 0
    assert args.resume is None or os.path.exists(args.resume)

    dataset, num_classes = build_dataset(image_set="val", config=main_args, num_videos_to_eval=main_args.val_videos_eval)
    sampler_val = torch.utils.data.SequentialSampler(dataset)

    data_loader_val = DataLoader(
        dataset, 1,
        sampler=sampler_val,
    collate_fn=val_collate, num_workers=0)

    model, _, _ = build_model(num_classes, main_args)
    model.cuda()
    model.eval()
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])

    t = measure_average_inference_time(model, data_loader_val, main_args.num_frames, args.num_iters, args.warm_iters)
    return 1.0 / t * args.batch_size


if __name__ == '__main__':
    fps = benchmark()
    print(f'Inference Speed: {fps:.1f} FPS')

