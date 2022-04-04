'''
Inference code for VisTR
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from datasets import build_dataset
from models import build_model
import os
import torch.nn.functional as F
import json
from util.viz_utils import visualize_tracks_independently, visualize_clips_after_processing, visualize_results_merged
import pycocotools.mask as mask_util
import trackeval
import tqdm
from zipfile import ZipFile
import copy
from models.matcher import HungarianInferenceMatcher
import time


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--use_extra_class', action='store_true', help="Use DeformableVisTR")

    # * Hungarian Inference Matcher Coefficients
    parser.add_argument('--cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--cost_mask_iou', default=6, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--cost_score', default=0, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--cost_center_distance', default=0, type=float,
                        help="L1 box coefficient in the matching cost")

    # * Final Track computation
    parser.add_argument('--use_binary_mask_iou', action='store_true')
    parser.add_argument('--use_frame_average_iou', action='store_true')
    parser.add_argument('--use_center_distance', action='store_true')

    parser.add_argument('--top_k_inference', type=int, default=None)

    parser.add_argument('--final_class_policy', default='most_common', type=str, choices=('most_common', 'score_weighting'),)
    parser.add_argument('--final_score_policy', default='mean', type=str, choices=('mean', 'median'),)

    parser.add_argument('--track_min_detection_score', default=0.01, type=float, help="Number of query slots")
    parser.add_argument('--track_min_score', default=0.02, type=float, help="Number of query slots")
    parser.add_argument('--track_min_detections', default=1, type=int, help="Number of query slots")

    parser.add_argument('--viz_att_maps', action='store_true')


    # dataset parameters
    parser.add_argument('--transform_pipeline', default='optimized_transform')
    parser.add_argument('--transform_strategy', default='vistr')
    parser.add_argument('--max_size', default=800, type=int)
    parser.add_argument('--out_scale', default=1, type=float)
    parser.add_argument('--val_width', default=300, type=int)

    parser.add_argument('--dataset_type', default='vis')

    parser.add_argument('--data_path', default='/usr/prakt/p028/data')
    parser.add_argument('--train_set', default='yt_vis_train')
    parser.add_argument('--val_set', default='yt_vis_train')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evaluation parameters
    parser.add_argument('--overlap_window', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--save_result', default="eval_result")
    parser.add_argument('--out_viz_path', default="")
    parser.add_argument('--merge_tracks', action='store_true')
    parser.add_argument('--save_clip_viz',  action='store_true')
    parser.add_argument('--save_raw_detections',  action='store_true')

    #validation launch
    parser.add_argument('--input_folder', default="")
    parser.add_argument('--force_bug', action='store_true',)

    parser.add_argument('--epochs_to_eval', default=[6, 7, 8, 9, 10], type=int, nargs='+')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.input_folder:
        args.resume = os.path.join(args.input_folder, f"checkpoint_epoch_{args.epochs_to_eval[0]}.pth")
        args.save_result = f"val_epoch_{args.epochs_to_eval[0]}"

    state_dict = torch.load(args.resume)
    model_args = state_dict['args']

    args.focal_loss = model_args.focal_loss
    args.num_frames = model_args.num_frames

    # Check val_args
    # if model_args.transform_strategy != args.transform_strategy:
    #     raise ValueError("Transform strategy selected doesnt match the ones used for training")

    if model_args.max_size != args.max_size:
        print("WARNING: MAX_SIZE DIFFERENT THAN THE ONE USED FOR TRAINING")

    if model_args.out_scale != args.out_scale:
        print("WARNING: OUT SCALE DIFFERENT THAN THE ONE USED FOR TRAINING")

    if model_args.val_width != args.val_width:
        print("WARNING: VAL_WIDTH DIFFERENT THAN THE ONE USED FOR TRAINING")

    if model_args.overlap_window != args.overlap_window:
        print("WARNING: OVERLAP_WINDOW DIFFERENT THAN THE ONE USED FOR TRAINING")

    if model_args.top_k_inference != args.top_k_inference:
        print("WARNING: Using top_k for inference but was not used for training validation")
        model_args.top_k_inference = args.top_k_inference

    dataset_val, num_classes = build_dataset("val", args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(
        dataset_val, 1,
        sampler=sampler_val,
    collate_fn=utils.val_collate, num_workers=0)

    model, criterion, postprocessors = build_model(num_classes, model_args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    model_without_ddp.load_state_dict(state_dict['model'])

    model.eval()
    run_inference(model, data_loader_val, device, args)
    if args.input_folder and len(args.epochs_to_eval) > 1:
        for epoch_to_eval in args.epochs_to_eval[1:]:
            print(f"*********************** Starting validation epoch {epoch_to_eval} ***********************")
            checkpoint_path = os.path.join(args.input_folder, f"checkpoint_epoch_{epoch_to_eval}.pth")
            assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} DON'T EXISTS"
            args.save_result = f"val_epoch_{epoch_to_eval}"
            state_dict = torch.load(checkpoint_path)
            model_without_ddp.load_state_dict(state_dict['model'])
            model.eval()
            run_inference(model, data_loader_val, device, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
