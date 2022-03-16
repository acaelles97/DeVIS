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
from util.viz_utils import visualize_tracks_independently, visualize_raw_frame_detections, visualize_clips_after_processing
import pycocotools.mask as mask_util
import trackeval
import tqdm
from zipfile import ZipFile
import copy
import time
from scipy.optimize import linear_sum_assignment
from typing import  List


INF = 100000000

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
    parser.add_argument('--save_clip_viz',  action='store_true')
    parser.add_argument('--save_raw_detections',  action='store_true')

    #validation launch
    parser.add_argument('--input_folder', default="")
    parser.add_argument('--epochs_to_eval', default=[6,7,8,9,10], type=int, nargs='+')
    return parser


def evaluate_ovis_accums(acum_results, gt_path, out_folder):
    # Eval config
    eval_config = trackeval.Evaluator.get_default_eval_config()
    # print only combined since TrackMAP is undefined for per sequence breakdowns
    eval_config['PRINT_ONLY_COMBINED'] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["OUTPUT_DETAILED"] = False
    eval_config["PLOT_CURVES"] = False
    eval_config["LOG_ON_ERROR"] = False

    # Dataset config
    dataset_config = trackeval.datasets.YouTubeVIS.get_default_dataset_config()
    dataset_config["PRINT_CONFIG"] = False

    dataset_config["OUTPUT_FOLDER"] = out_folder
    dataset_config["TRACKER_DISPLAY_NAMES"] = ["DefTrackFormer"]
    dataset_config["TRACKERS_TO_EVAL"] = ["DefTrackFormer"]

    # Metrics config
    metrics_config = {'METRICS': ['TrackMAP']}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.YouTubeVIS(dataset_config, gt=gt_path, predictions=acum_results)]
    metrics_list = []
    for metric in [trackeval.metrics.TrackMAP, trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                   trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            # specify TrackMAP config for YouTubeVIS
            if metric == trackeval.metrics.TrackMAP:
                default_track_map_config = metric.get_default_metric_config()
                default_track_map_config['USE_TIME_RANGES'] = False
                default_track_map_config['AREA_RANGES'] = [[0 ** 2, 128 ** 2],
                                                           [128 ** 2, 256 ** 2],
                                                           [256 ** 2, 1e5 ** 2]]

                default_track_map_config['MAX_DETECTIONS'] = 100
                metrics_list.append(metric(default_track_map_config))
            else:
                metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    eval_results, eval_msg = evaluator.evaluate(dataset_list, metrics_list)
    clas_av = eval_results['YouTubeVIS']['DefTrackFormer']['COMBINED_SEQ']['cls_comb_cls_av']['TrackMAP']
    # Mean score for each  Iou th from 0.5::0.95
    class_av_ap_all, class_av_ar_all = 100*np.mean(clas_av["AP_all"]), 100*np.mean(clas_av["AR_all"])
    det_av = eval_results['YouTubeVIS']['DefTrackFormer']['COMBINED_SEQ']['cls_comb_det_av']['TrackMAP']
    # det_av_ap_all, det_av_ar_all = 100*np.mean(det_av["AP_all"]), 100*np.mean(det_av["AR_all"])

    return class_av_ap_all, class_av_ar_all


class Track:
    def __init__(self, track_id, track_length, start_idx=0):
        self._id = track_id
        self.length = track_length
        self.start_idx = start_idx
        self.attrs = ["scores", "masks", "categories", "boxes", "centroid_points"]
        self.scores = []
        self.masks = []
        self.categories = []
        self.boxes = []
        self.centroid_points = []
        self.valid_frames = []
        self.mask_id = None
        self.last_t = 0

    def __hash__(self):
        return self._id

    def __len__(self):
        return self.length

    def __eq__(self, other):
        return self._id == hash(other)

    def get_id(self):
        return self._id

    def valid(self, min_detections= 1):
        return sum(self.valid_frames) >= min_detections

    def process_masks(self, idx, t_window, tgt_size, encode_all_mask, masks):
        num_masks = masks.shape[0]
        processed_masks = []
        for t in range(len(self.categories)):
            mask = masks[t]
            mask = F.interpolate(mask[None, None], tgt_size, mode="bilinear").detach().sigmoid()[0, 0]
            if encode_all_mask:
                processed_masks.append(self.encode_mask(mask))
            else:
                if idx == 0:
                    if t < num_masks - t_window:
                        processed_masks.append(self.encode_mask(mask))
                    else:
                        processed_masks.append(mask)
                else:
                    if t_window + self.start_idx <= t < num_masks - t_window or t < self.start_idx:
                        processed_masks.append(self.encode_mask(mask))
                    else:
                        processed_masks.append(mask)
        self.masks = processed_masks

    def update(self, idx, t_window, tgt_size,  scores, categories, boxes, masks, centroid_points, top_k, mask_id, encode_all_mask):
        self.scores = scores.tolist()
        self.categories = categories.tolist()
        self.boxes = boxes.tolist()
        self.centroid_points = centroid_points.tolist()
        self.mask_id = mask_id

        self.masks = masks

        # if top_k and isinstance(masks, list):
        #
        # else:
        #     self.process_masks(idx, t_window, tgt_size, encode_all_mask, masks)

    def init_video_track(self, num_t, track):
        for attr in self.attrs:
            getattr(self, attr)[:num_t] = getattr(track, attr)
        self.mask_id = track.mask_id

    def valid_for_matching_end(self, t_window):
        return not (any(score is None for score in self.scores[self.last_t-t_window:self.last_t]) or any(mask is None for mask in self.masks[self.last_t-t_window:self.last_t]))

    def valid_for_matching_start(self, t_window):
        return not (any(score is None for score in self.scores[self.start_idx:(self.start_idx+t_window)]) or any(mask is None for mask in self.masks[self.start_idx:(self.start_idx+t_window)]))

    def mean_score(self):
        scores = [score for valid, score in zip(self.valid_frames, self.scores) if valid]
        if not scores:
            return 0
        else:
            return np.mean(scores)

    def median_score(self):
        scores = [score for valid, score in zip(self.valid_frames, self.scores) if valid]
        if not scores:
            return 0
        else:
            return np.median(scores)

    def get_last_t_result(self, t, attr):
        return getattr(self, attr)[self.last_t+t]

    def get_last_results(self, t_window, attr):
        return getattr(self, attr)[self.last_t-t_window: self.last_t]

    def get_mask_id(self):
        return self.mask_id

    def get_first_t_result(self, t, attr):
        return getattr(self, attr)[(self.start_idx+t)]

    def get_first_results(self, t_window, attr):
        return getattr(self, attr)[self.start_idx: self.start_idx+t_window]

    def get_results_to_append(self, t, attr):
        return getattr(self, attr)[(self.start_idx + t):]

    def get_results_to_start(self, attr):
        return getattr(self, attr)

    def append_track(self, track, t_window: int):
        # Overwrite all current detections for the overlap frames
        overlap_positions = range(self.last_t-t_window-track.start_idx, self.last_t)
        for other_pos, self_pos in enumerate(overlap_positions):
            other_score = track.scores[other_pos] if track.scores[other_pos] is not None else 0
            self_score = self.scores[self_pos] if self.scores[self_pos] is not None else 0
            if other_score > self_score:
                for attr in self.attrs:
                    if attr == "masks":
                        other_mask = track.masks[other_pos]
                        if not isinstance(other_mask, dict):
                            other_mask = self.encode_mask(other_mask)
                        self.masks[self_pos] = other_mask
                    else:
                        getattr(self, attr)[self_pos] = getattr(track, attr)[other_pos]
            else:
                self_mask = self.masks[self_pos]
                if not isinstance(self_mask, dict):
                    self_mask = self.encode_mask(self_mask)

                self.masks[self_pos] = self_mask

        for attr in self.attrs:
            results = track.get_results_to_append(t_window, attr)
            getattr(self, attr)[self.last_t:self.last_t+len(results)] = results

        self.mask_id = track.mask_id

    def update_curr_timestep(self, t):
        self.last_t += t

    def filter_frame_detections(self, min_detection_score):
        # Eliminates individual detections with score < min_detection_score
        for idx, score in enumerate(self.scores):
            if  score < min_detection_score:
                self.valid_frames.append(False)
            else:
                self.valid_frames.append(True)


    def compute_final_score(self, policy):
        if policy == "mean":
            return self.mean_score()
        elif policy == "median":
            return self.median_score()
        else:
            raise ValueError("Score policy not implemented: Available: mean, median")

    def compute_final_category(self, policy):
        if policy == "most_common":
            category_ids = np.array([cat for valid, cat in zip(self.valid_frames, self.categories) if valid])
            return np.argmax(np.bincount(category_ids))

        elif policy == "score_weighting":
            score_dict = {}
            for valid, score, cat in zip(self.valid_frames, self.scores, self.categories):
                if not valid:
                    continue
                if cat not in score_dict:
                    score_dict[cat] = 0
                score_dict[cat] += score
            values, idx_dict = [], {}
            for idx, (key, value) in enumerate(score_dict.items()):
                values.append(value)
                idx_dict[idx] = key
            cat_idx = np.array(values).argmax()
            return idx_dict[cat_idx]

        else:
            raise ValueError("Category policy not implemented: Available: most_common, score_weighting")


    def get_formatted_result(self, video_id, category_policy, score_policy):
        final_score = self.compute_final_score(score_policy)
        final_category_id = self.compute_final_category(category_policy)
        final_masks = []
        for valid, mask in zip(self.valid_frames, self.masks):
            if valid:
                final_masks.append(mask)
            else:
                final_masks.append(None)

        track = {'video_id': int(video_id),
                'score': float(final_score),
                'category_id': int(final_category_id),
                'segmentations': final_masks}

        return track


    def encode_initial_masks(self, t_widow):
        for t in range(0, self.last_t-t_widow):
            mask = self.masks[t].numpy() > 0.5
            rle_mask = mask_util.encode(np.array(mask, order='F'))
            rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
            self.masks[t] = rle_mask

    def encode_mask(self, mask):
        mask = mask.cpu().numpy() > 0.5
        rle_mask = mask_util.encode(np.array(mask, order='F'))
        rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
        return rle_mask

    def encode_all_masks(self):
        for t in range(self.length):
            if self.masks[t] is not None and not isinstance(self.masks[t], dict):
                self.masks[t] = self.encode_mask(self.masks[t])



def process_mask(masks, tgt_size):
    # mask = masks.cpu() > 0.5
    mask = F.interpolate(masks, tgt_size, mode="bilinear", align_corners=False).cpu() > 0.5
    mask = mask[:, 0].unsqueeze(-1).transpose(0, -1)[0].numpy()
    rle_mask = mask_util.encode(np.array(mask, order='F'))
    for mask in rle_mask:
        mask["counts"] = mask["counts"].decode("utf-8")
    return rle_mask




class ClipResultsMatching:

    def __init__(self, results, start_idx, overlap_window, tgt_video_size):
        self.overlap_window = overlap_window
        self.scores = results["scores"]
        self.classes = results["labels"]
        self.boxes = results["boxes"]
        # self.masks = F.interpolate(results["masks"], tgt_video_size, mode="bilinear", align_corners=False).sigmoid()
        self.masks = results["masks"].sigmoid()
        self.centroid_points = results["centroid_points"]
        self.real_idx = results['inverse_idxs']
        self.start_idx = start_idx
        self.tgt_video_size = tgt_video_size

        num_frames = self.classes.shape[0]
        if num_frames - (start_idx + overlap_window) < overlap_window:
            self.end_elements = num_frames - (start_idx + overlap_window)
        else:
            self.end_elements = overlap_window
        self.ordered_ids = None

        self.other_mask_indices = None
        self.other_score_mask = None
        self.other_masks = None
        self.other_centroids = None

    def order_results(self, ordered_ids):
        self.scores = self.scores[:, ordered_ids]
        self.classes = self.classes[:, ordered_ids]
        self.boxes = self.boxes[:, ordered_ids]
        self.real_idx = self.real_idx[ordered_ids]

    def get_num_instances(self):
        return self.scores.shape[0]

    def get_result_to_match(self, attr):
        return getattr(self, attr)[self.start_idx: self.start_idx + self.overlap_window]


    def add_matching_clip_results(self, other):

        other_overlap_positions =  torch.arange(0, other.overlap_window)
        self_overlap_positions = torch.arange(self.start_idx, self.start_idx + self.overlap_window)

        mask_ = other.scores[other_overlap_positions] > self.scores[self_overlap_positions]

        # We can not update mask info
        self.scores[self_overlap_positions][mask_] = other.scores[other_overlap_positions][mask_]
        self.boxes[self_overlap_positions][mask_] = other.boxes[other_overlap_positions][mask_]
        self.classes[self_overlap_positions][mask_] = other.classes[other_overlap_positions][mask_]

        self.other_mask_indices = other.real_idx
        self.other_score_mask = mask_
        self.other_masks = other.masks
        self.other_centroids = other.centroid_points


    def remove_matched_data(self):
        self.scores = self.scores[-self.end_elements:]
        self.classes = self.classes[-self.end_elements:]
        self.boxes = self.boxes[-self.end_elements:]
        self.masks = self.masks[-self.end_elements:]
        self.centroid_points = self.centroid_points[-self.end_elements:]


class Tracker:

    def __init__(self, model, args):
        self.model = model
        self.hungarian_matcher = HungarianInferenceMatcher(cost_mask_iou=args.cost_mask_iou , cost_class=args.cost_class,
                                                            t_window=args.overlap_window, score_cost=args.cost_score, cost_center_distance=args.cost_center_distance)
        self.focal_loss = args.focal_loss
        self.num_frames = args.num_frames
        self.tracker_cfg = args.tracker_cfg
        self.overlap_window = args.overlap_window
        self.top_k =  args.top_k_inference


    def update_video_start(self, video, clip_results, tgt_video_size):

        tmp_indexes = clip_results.real_idx.cpu().tolist()

        processed_masks = {}
        for idx, track in enumerate(video):
            track.scores.extend(clip_results.scores[:-self.overlap_window, idx].cpu().tolist())
            track.categories.extend(clip_results.classes[:-self.overlap_window, idx].cpu().tolist())
            track.boxes.extend(clip_results.boxes[:-self.overlap_window, idx].cpu().tolist())

            if tmp_indexes[idx] not in processed_masks:
                masks_to_process =  clip_results.masks[:-self.overlap_window, tmp_indexes[idx]][:, None]
                processed_mask = process_mask(masks_to_process, tgt_video_size)
                processed_masks[tmp_indexes[idx]] = processed_mask
            else:
                processed_mask = processed_masks[tmp_indexes[idx]]

            track.masks.extend(processed_mask)
            track.centroid_points.extend(clip_results.centroid_points[:-self.overlap_window, tmp_indexes[idx]].cpu().tolist())

    
    def update_video_in_between(self, videos, clip_results, tgt_video_size, matching_indices):

        mask_id_indices_clip = clip_results.real_idx.cpu().tolist()
        mask_id_indices_from_other = clip_results.other_mask_indices.cpu().tolist()

        clip_processed_masks, matching_window_processed_masks = {}, {}

        for idx, video_index in enumerate(matching_indices):
            track = videos[video_index]
            track.scores.extend(clip_results.scores[clip_results.start_idx:-clip_results.end_elements, idx].cpu().tolist())
            track.categories.extend(clip_results.classes[clip_results.start_idx:-clip_results.end_elements, idx].cpu().tolist())
            track.boxes.extend(clip_results.boxes[clip_results.start_idx:-clip_results.end_elements, idx].cpu().tolist())

            # Create and store masks from the current clip
            mask_id = mask_id_indices_clip[idx]
            if mask_id not in clip_processed_masks:
                masks_to_process = clip_results.masks[clip_results.start_idx:-clip_results.end_elements, mask_id][:, None]
                processed_mask_clip = process_mask(masks_to_process, tgt_video_size)
                clip_processed_masks[mask_id] = processed_mask_clip
            else:
                processed_mask_clip = clip_processed_masks[mask_id]

            if torch.any(clip_results.other_score_mask[:, idx]):
                final_masks = []
                final_centroids = []

                # Create and store masks from the previous clip on the overlap window
                mask_id_overlap = mask_id_indices_from_other[idx]
                if mask_id_overlap not in matching_window_processed_masks:
                    masks_to_process = clip_results.other_masks[:, mask_id_overlap][:, None]
                    processed_mask_overlap = process_mask(masks_to_process, tgt_video_size)
                    matching_window_processed_masks[mask_id_overlap] = processed_mask_overlap

                else:
                    processed_mask_overlap = matching_window_processed_masks[mask_id_overlap]

                # Merge masks from the previous clip and the current clip on the overlap window, picking the one with highest score
                for tmp_idx_overlap, pick_other_mask in enumerate(clip_results.other_score_mask[:, idx]):
                    if pick_other_mask:
                        final_masks.append(processed_mask_overlap[tmp_idx_overlap])
                        final_centroids.append(clip_results.other_centroids[tmp_idx_overlap, mask_id_overlap].cpu().tolist())
                    else:
                        final_masks.append(processed_mask_clip[tmp_idx_overlap])
                        final_centroids.append(clip_results.centroid_points[clip_results.start_idx + tmp_idx_overlap, mask_id].cpu().tolist())

                # Add the final masks that are not in the past overlap window nether on the future overlap window
                final_masks.extend(processed_mask_clip[self.overlap_window:])
                track.masks.extend(final_masks)
                final_centroids.extend(clip_results.centroid_points[clip_results.start_idx + self.overlap_window:-clip_results.end_elements, mask_id].cpu().tolist())
                track.centroid_points.extend(final_centroids)


            else:
                track.masks.extend(processed_mask_clip)
                track.centroid_points.extend(clip_results.centroid_points[clip_results.start_idx:-clip_results.end_elements, mask_id].cpu().tolist())


    def update_video_end(self, videos, clip_results, tgt_video_size, matching_indices):
        tmp_indexes = clip_results.real_idx.cpu().tolist()

        processed_masks = {}
        for idx, video_index in enumerate(matching_indices):
            track = videos[video_index]

            track.scores.extend(clip_results.scores[:, idx].cpu().tolist())
            track.categories.extend(clip_results.classes[:, idx].cpu().tolist())
            track.boxes.extend(clip_results.boxes[:, idx].cpu().tolist())

            if tmp_indexes[idx] not in processed_masks:
                masks_to_process =  clip_results.masks[:, tmp_indexes[idx]][:, None]
                processed_mask = process_mask(masks_to_process, tgt_video_size)
                processed_masks[tmp_indexes[idx]] = processed_mask
            else:
                processed_mask = processed_masks[tmp_indexes[idx]]

            track.masks.extend(processed_mask)
            track.centroid_points.extend(clip_results.centroid_points[:, tmp_indexes[idx]].cpu().tolist())


    @torch.no_grad()
    def __call__(self, video, device, args):
        sampler_val = torch.utils.data.SequentialSampler(video)
        video_loader = DataLoader(video, 1, sampler=sampler_val, num_workers=args.num_workers)
        real_video_length = video.real_video_length
        clip_length = self.num_frames if real_video_length is None or real_video_length >= self.num_frames else real_video_length
        cat_names = video.cat_names
        video_info = {
            "tgt_size": video.original_size,
            "clip_length": clip_length
        }

        clip_results_to_match = None
        video_tracks = [Track(track_id, video.final_video_length, 0) for track_id in range(self.top_k)]

        matched_tracks1_ids = list(range(0, self.top_k))
        for idx, video_clip in enumerate(video_loader):
            video_clip = video_clip.to(device)
            results = self.model(video_clip.squeeze(0), video_info)
            start_idx = 0 if idx != len(video_loader) - 1  else video.last_real_idx
            if clip_results_to_match is None:
                clip_results_to_match = ClipResultsMatching(results, start_idx, self.overlap_window, video.original_size)
                self.update_video_start(video_tracks, clip_results_to_match, video.original_size)
                clip_results_to_match.remove_matched_data()

            else:
                clip_results = ClipResultsMatching(results, start_idx, self.overlap_window, video.original_size)
                matched_tracks1_ids, matched_tracks2_ids = self.hungarian_matcher(clip_results_to_match, clip_results)

                clip_results_to_match.order_results(torch.tensor(matched_tracks1_ids))
                clip_results.order_results(torch.tensor(matched_tracks2_ids))

                clip_results.add_matching_clip_results(clip_results_to_match)

                # We need to reorder video with matched_tracks1_ids
                self.update_video_in_between(video_tracks, clip_results,  video.original_size, matched_tracks1_ids)

                clip_results.remove_matched_data()
                clip_results_to_match = clip_results


        self.update_video_end(video_tracks, clip_results_to_match, video.original_size, matched_tracks1_ids)

        if self.tracker_cfg.track_min_detection_score != 0:
            for track in video_tracks:
                track.filter_frame_detections(self.tracker_cfg.track_min_detection_score)

        keep = np.array([track.valid(min_detections=self.tracker_cfg.track_min_detections) for track in video_tracks])
        video_tracks = [track for i, track in enumerate(video_tracks) if keep[i]]

        if self.tracker_cfg.track_min_score != 0:
            keep = [track.compute_final_score(self.tracker_cfg.final_score_policy) > self.tracker_cfg.track_min_score for track in video_tracks]
            video_tracks = [track for i, track in enumerate(video_tracks) if keep[i]]

        final_tracks_result = [track.get_formatted_result(video.video_id, self.tracker_cfg.final_class_policy, self.tracker_cfg.final_score_policy) for track in video_tracks]

        return final_tracks_result



class HungarianInferenceMatcher:

    def __init__(self, t_window: int = 2, cost_class: float = 2, cost_mask_iou: float = 6, score_cost: float = 2, cost_center_distance = 0):
        self.t_window = t_window
        self.class_cost = -1 * cost_class
        self.mask_iou_cost = -1 * cost_mask_iou
        self.score_cost = score_cost
        self.cost_center_distance = cost_center_distance

    @staticmethod
    def compute_class_cost(clip_results1: ClipResultsMatching, clip_results2: ClipResultsMatching):
        classes_clip1 = clip_results1.classes.transpose(0, 1)[:, None]
        classes_clip2 =  clip_results2.get_result_to_match("classes").transpose(0, 1)[None]

        total_cost_classes = classes_clip1 == classes_clip2
        total_cost_classes = total_cost_classes.type(torch.float32).mean(-1)

        return total_cost_classes

    @staticmethod
    def compute_score_cost(clip_results1: ClipResultsMatching, clip_results2: ClipResultsMatching):
        scores_clip1 = clip_results1.scores.transpose(0, 1)[:, None]
        scores_clip2 = clip_results2.get_result_to_match("scores").transpose(0, 1)[None]

        total_cost_scores = torch.abs(scores_clip1 - scores_clip2).mean(-1)
        return total_cost_scores

    @staticmethod
    def compute_center_distance_cost(clip_results1: ClipResultsMatching, clip_results2: ClipResultsMatching):
        centroids_clip1 = clip_results1.centroid_points.transpose(0, 1)[:, None]
        centroids_clip2 = clip_results2.get_result_to_match("centroid_points").transpose(0, 1)[None]

        cost_ct_distance = torch.abs(centroids_clip1 - centroids_clip2).sum(-1).mean(-1)

        top_k = clip_results1.scores.shape[1]
        clip1_idx = clip_results1.real_idx[:, None].repeat(1, top_k).flatten(0, 1)
        clip2_idx = clip_results2.real_idx.repeat(top_k)

        total_cost_ct_distance = cost_ct_distance[clip1_idx, clip2_idx].view(top_k, top_k)
        return total_cost_ct_distance

    @staticmethod
    def compute_volumetric_siou_cost(clip_results1: ClipResultsMatching, clip_results2: ClipResultsMatching):
        # Inspiration from IFC https://github.com/sukjunhwang/IFC
        masks_clip1 = clip_results1.masks.transpose(0, 1).flatten(-2)[:, None]
        masks_clip2 = clip_results2.get_result_to_match("masks").transpose(0, 1).flatten(-2)[None]

        numerator = masks_clip1 * masks_clip2
        denominator = masks_clip1 + masks_clip2 - masks_clip1 * masks_clip2

        numerator = numerator.sum(dim=(-1, -2))
        denominator = denominator.sum(dim=(-1, -2))

        siou = numerator / (denominator + 1e-6)

        top_k =  clip_results1.scores.shape[1]
        clip1_idx = clip_results1.real_idx[:, None].repeat(1, top_k).flatten(0, 1)
        clip2_idx = clip_results2.real_idx.repeat(top_k)

        total_cost_siou = siou[clip1_idx,  clip2_idx].view(top_k, top_k)

        return total_cost_siou

    def __call__(self, clip_results1, clip_results2):
        total_cost_iou = self.compute_volumetric_siou_cost(clip_results1, clip_results2)
        total_cost_classes = self.compute_class_cost(clip_results1, clip_results2)
        total_cost_scores = self.compute_score_cost(clip_results1, clip_results2)
        total_cost_ct = self.compute_center_distance_cost(clip_results1, clip_results2)

        cost =  total_cost_iou * self.mask_iou_cost + total_cost_classes * self.class_cost + total_cost_scores * self.score_cost + self.cost_center_distance * total_cost_ct

        cost = cost.cpu().numpy()
        track1_ids, track2_ids = linear_sum_assignment(cost)


        return track1_ids, track2_ids


def run_inference(model, data_loader_val, device, args):
    MB = 1024.0 * 1024.0
    tracker_cfg = {
        "track_min_detection_score": args.track_min_detection_score,
        "track_min_score": args.track_min_score,
        "track_min_detections": args.track_min_detections,
        "final_class_policy": args.final_class_policy,
        "final_score_policy": args.final_score_policy,
    }

    all_tracks = []
    args.tracker_cfg = utils.nested_dict_to_namespace(tracker_cfg)
    tracker = Tracker(model, args)
    init_time = time.time()
    for idx, video in tqdm.tqdm(enumerate(data_loader_val)):
        print(f" Max memory allocated {int(torch.cuda.max_memory_allocated() / MB)}")
        video_tracks = tracker(video, device, args)
        if args.out_viz_path:
            visualize_tracks_independently(video.images_folder, video.video_clips, video_tracks, out_path=args.out_viz_path,
                                           class_name=data_loader_val.dataset.cat_names)
        all_tracks.extend(video_tracks)
    finish_time = time.time()
    print(f"Total time {finish_time-init_time}")

    gathered_preds = utils.all_gather(all_tracks)
    results_accums = utils.accumulate_results(gathered_preds)

    print("FINISHED")
    class_av_ap_all, class_av_ar_all = None, None
    if data_loader_val.dataset.has_gt:
        out_eval_folder = os.path.join(args.output_dir, "output_eval")
        if not os.path.exists(out_eval_folder):
            os.makedirs(out_eval_folder, exist_ok=True)
        class_av_ap_all, class_av_ar_all = evaluate_ovis_accums(results_accums, data_loader_val.dataset.annotations, out_eval_folder)

    if args.save_result or not data_loader_val.dataset.has_gt:
        out_dir = os.path.join(args.output_dir, args.save_result)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "results.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results_accums, f)

        out_zip_filename = os.path.join(out_dir, "results.zip")
        with ZipFile(out_zip_filename, 'w') as zip_obj:
            zip_obj.write(out_file, os.path.basename(out_file))

    return class_av_ap_all, class_av_ar_all

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
    if not hasattr(model_args, "balance_aux_loss"):
        model_args.balance_aux_loss = False
    if not hasattr(model_args, "trajectory_loss"):
        model_args.trajectory_loss = False
    if not hasattr(model_args, "top_k_inference"):
        model_args.top_k_inference = None
    if not hasattr(model_args, "volumetric_mask_loss"):
        model_args.volumetric_mask_loss = None
    if not hasattr(model_args, "mask_aux_loss"):
        model_args.mask_aux_loss = None
    if not hasattr(model_args, "use_trajectory_queries"):
        model_args.use_trajectory_queries = False
    if not hasattr(model_args, "use_extra_class"):
        model_args.use_extra_class = False
    if not hasattr(model_args, "use_giou"):
        model_args.use_giou = False
    if not hasattr(model_args, "use_l1_distance_sum"):
        model_args.use_l1_distance_sum = False
    if not hasattr(model_args, "create_bbx_from_mask"):
        model_args.create_bbx_from_mask = False
    if not hasattr(model_args, "use_non_valid_class"):
        model_args.use_non_valid_class = False
    if not hasattr(model_args, "use_instance_level_classes"):
        model_args.use_instance_level_classes = False
    if not hasattr(model_args, "mask_attn_alignment"):
        model_args.mask_attn_alignment = False
    if not hasattr(model_args, "use_box_coords"):
        model_args.use_box_coords = False
    if not hasattr(model_args, "class_head_type"):
        model_args.class_head_type = None
    if not hasattr(model_args, "balance_mask_aux_loss"):
        model_args.balance_mask_aux_loss = False
    if not hasattr(model_args, "with_gradient"):
        model_args.with_gradient = False
    if not hasattr(model_args, "with_single_class_embed"):
        model_args.with_single_class_embed = False
    if not hasattr(model_args, "with_decoder_frame_self_attn"):
        model_args.with_decoder_frame_self_attn = False
    if not hasattr(model_args, "with_decoder_instance_self_attn"):
        model_args.with_decoder_instance_self_attn = False

    if not hasattr(model_args, "with_class_inst_attn"):
        model_args.with_class_inst_attn = False

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