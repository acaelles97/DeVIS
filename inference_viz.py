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
    class_av_ap_all, class_av_ar_all = 100 * np.mean(clas_av["AP_all"]), 100*np.mean(clas_av["AR_all"])
    det_av = eval_results['YouTubeVIS']['DefTrackFormer']['COMBINED_SEQ']['cls_comb_det_av']['TrackMAP']
    # det_av_ap_all, det_av_ar_all = 100*np.mean(det_av["AP_all"]), 100*np.mean(det_av["AR_all"])

    return class_av_ap_all, class_av_ar_all


class Track:
    def __init__(self, track_id, track_length, start_idx=0):
        self._id = track_id
        self.length = track_length
        self.start_idx = start_idx
        self.attrs = ["scores", "masks", "categories", "boxes", "centroid_points"]
        self.scores = [None for _ in range(track_length)]
        self.masks = [None for _ in range(track_length)]
        self.categories = [None for _ in range(track_length)]
        self.boxes = [None for _ in range(track_length)]
        self.centroid_points = [None for _ in range(track_length)]
        self.valid_frames = [True for _ in range(track_length)]
        self.mask_id = None
        self.last_t = 0
        self.matching_ids_record = []

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

    def update(self, scores, categories, boxes, masks, centroid_points, mask_id):
        self.scores = scores.tolist()
        self.categories = categories.tolist()
        self.boxes = boxes.tolist()
        self.centroid_points = centroid_points.tolist()
        self.mask_id = mask_id
        self.masks = masks

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
        self.matching_ids_record.append((self._id, track.get_id()))
        inference_clip_size = track.length
        overlap_positions = range(self.last_t-t_window-track.start_idx, self.last_t)
        for other_pos, self_pos in enumerate(overlap_positions):
            other_score = track.scores[other_pos] if track.scores[other_pos] is not None else 0
            self_score = self.scores[self_pos] if self.scores[self_pos] is not None else 0
            if other_score > self_score:
                for attr in self.attrs:
                    if attr == "masks":
                        other_mask = track.masks[other_pos]
                        # if not isinstance(other_mask, dict):
                        #     other_mask = encode_mask(other_mask)
                        self.masks[self_pos] = other_mask
                    else:
                        getattr(self, attr)[self_pos] = getattr(track, attr)[other_pos]
            # else:
                # self_mask = self.masks[self_pos]
                # if not isinstance(self_mask, dict):
                #     self_mask = encode_mask(self_mask)
                #
                # self.masks[self_pos] = self_mask

        for attr in self.attrs:
            results = track.get_results_to_append(t_window, attr)
            getattr(self, attr)[self.last_t:self.last_t+len(results)] = results

        self.mask_id = track.mask_id

    def update_stride(self, stride):
        self.last_t += stride

    def update_stride_and_encode_masks(self, stride, overlap_window):
        # Check if mask can be encoded and save
        for idx in range(self.last_t-overlap_window, self.last_t-overlap_window + stride):
            if 0 <= idx < len(self.masks) and not isinstance(self.masks[idx], dict):
                self.masks[idx] = encode_mask(self.masks[idx])

        self.last_t += stride


    def filter_frame_detections(self, min_detection_score):
        # Eliminates individual detections with score < min_detection_score
        for idx, score in enumerate(self.scores):
            if  score < min_detection_score:
                self.valid_frames[idx] = False

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


    def encode_all_masks(self):
        for t in range(self.length):
            if self.masks[t] is not None and not isinstance(self.masks[t], dict):
                self.masks[t] = encode_mask(self.masks[t])

    def process_centroid(self, tgt_size):
        img_h, img_w = torch.tensor([tgt_size]).unbind(-1)
        scale_fct = torch.stack([img_w, img_h], dim=1)
        new_centroids = []
        for centroid_p in self.centroid_points:
            if centroid_p is not None:
                centroid_points =  torch.tensor(centroid_p)[None] * scale_fct
                centroid_points[:, 0] = torch.clamp(centroid_points[:, 0], 0, img_w.item())
                centroid_points[:, 1] = torch.clamp(centroid_points[:, 1], 0, img_h.item())
                new_centroids.append(centroid_points.tolist()[0])
            else:
                new_centroids.append(None)

        self.centroid_points = new_centroids


def encode_mask(mask):
    mask = mask.cpu().numpy() > 0.5
    rle_mask = mask_util.encode(np.array(mask, order='F'))
    rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
    return rle_mask

class Tracker:

    def __init__(self, model, args):
        self.model = model
        self.hungarian_matcher = HungarianInferenceMatcher(cost_mask_iou=args.cost_mask_iou , cost_class=args.cost_class,
                                                            t_window=args.overlap_window, score_cost=args.cost_score, cost_center_distance=args.cost_center_distance,
                                                           use_binary_mask_iou=args.use_binary_mask_iou,
                                                           use_frame_average_iou=args.use_frame_average_iou, use_center_distance=args.use_center_distance)
        self.focal_loss = args.focal_loss
        self.num_frames = args.num_frames
        self.tracker_cfg = args.tracker_cfg
        self.overlap_window = args.overlap_window
        self.top_k_mode =  args.top_k_inference is not None

    def process_masks(self, start_idx, idx, tgt_size, encode_all_mask, masks):
        processed_masks = []
        num_masks = masks.shape[0]
        for t in range(num_masks):
            mask = masks[t]
            mask = F.interpolate(mask[None, None], tgt_size, mode="bilinear", align_corners=False).detach().sigmoid()[0, 0]
            if encode_all_mask:
                processed_masks.append(encode_mask(mask))

            else:
                if idx == 0:
                    if t < num_masks - self.overlap_window:
                        processed_masks.append(encode_mask(mask))
                    else:
                        processed_masks.append(mask)
                else:
                    if self.overlap_window + start_idx <= t < num_masks - self.overlap_window or t < start_idx:
                        processed_masks.append(encode_mask(mask))
                    else:
                        processed_masks.append(mask)
        return processed_masks


    @torch.no_grad()
    def __call__(self, video, device, all_times, args):
        sampler_val = torch.utils.data.SequentialSampler(video)
        video_loader = DataLoader(video, 1, sampler=sampler_val, num_workers=args.num_workers)
        real_video_length = video.real_video_length
        clip_length = self.num_frames if real_video_length is None or real_video_length >= self.num_frames else real_video_length
        cat_names = video.cat_names
        video_tracks = None
        video_info = {
            "tgt_size": video.original_size,
            "clip_length": clip_length
        }
        times = []

        for idx, video_clip in enumerate(video_loader):
            video_clip = video_clip.to(device)

            time1 = time.time()
            results = self.model(video_clip.squeeze(0), video_info)
            time_inference = time.time() - time1
            times.append(time_inference)

            results["centroid_prediction"] = True
            pred_scores, pred_classes, pred_boxes, pred_masks, pred_centroid_points  = results["scores"], results["labels"], results["boxes"], results["masks"], results["centroid_points"]
            detected_instances = pred_scores.shape[1]

            start_idx = 0 if idx != len(video_loader) - 1  else video.last_real_idx
            clip_tracks = [Track(track_id, clip_length, start_idx) for track_id in range(detected_instances)]

            # Processing all the masks at the same time was giving problems
            processed_masks_dict = {}
            processed_centroids_dict = {}
            encode_all_masks = self.hungarian_matcher.use_binary_mask_iou

            for i, track in enumerate(clip_tracks):
                mask_id = results['inverse_idxs'][i].item()
                if mask_id not in processed_masks_dict.keys():
                    processed_masks_dict[mask_id] = self.process_masks(start_idx, idx, video.original_size, encode_all_masks, pred_masks[:, mask_id])
                    processed_centroids_dict[mask_id] = pred_centroid_points[:, mask_id]

                track.update(pred_scores[:, i], pred_classes[:, i], pred_boxes[:, i], processed_masks_dict[mask_id],
                             processed_centroids_dict[mask_id], mask_id)
            time1 = time.time()

            if args.save_clip_viz and args.out_viz_path:
                clips_to_show = copy.deepcopy(clip_tracks)

                if self.tracker_cfg.track_min_detection_score != 0:
                    for track in clips_to_show:
                        track.filter_frame_detections(self.tracker_cfg.track_min_detection_score)
                keep = np.array([track.valid(min_detections=1) for track in clips_to_show])
                clips_to_show = [track for i, track in enumerate(clips_to_show) if keep[i]]

                if self.tracker_cfg.track_min_score != 0:
                    keep = [track.mean_score() > self.tracker_cfg.track_min_score for track in clips_to_show]
                    clips_to_show = [track for i, track in enumerate(clips_to_show) if keep[i]]
                for track in clips_to_show:
                    track.encode_all_masks()
                    track.process_centroid(video.original_size)

                visualize_clips_after_processing(idx, video.images_folder, video.video_clips[idx][:clip_length], clips_to_show, out_path=args.out_viz_path, class_name=cat_names)

            if video_tracks is None:
                video_tracks = [Track(track_id, video.final_video_length, start_idx) for track_id in range(detected_instances)]
                for new_track in clip_tracks:
                    video_tracks[new_track.get_id()].init_video_track(clip_length, new_track)
                for track in video_tracks:
                    track.update_stride(clip_length)

            else:
                matched_tracks1_ids, matched_tracks2_ids = self.hungarian_matcher(video_tracks, clip_tracks)
                for track_pos1, track_pos2 in zip(matched_tracks1_ids, matched_tracks2_ids):
                    video_tracks[track_pos1].append_track(clip_tracks[track_pos2], self.overlap_window)
                for track in video_tracks:
                    track.update_stride_and_encode_masks(clip_length-self.overlap_window, self.overlap_window)

            time_tracking = time.time() - time1
            times.append(time_tracking)


        print(f"FPS video { video.final_video_length / sum(times)}")
        all_times.append(sum(times))

        if self.tracker_cfg.track_min_detection_score != 0:
            for track in video_tracks:
                track.filter_frame_detections(self.tracker_cfg.track_min_detection_score)

        keep = np.array([track.valid(min_detections=self.tracker_cfg.track_min_detections) for track in video_tracks])
        video_tracks = [track for i, track in enumerate(video_tracks) if keep[i]]

        if self.tracker_cfg.track_min_score != 0:
            keep = [track.compute_final_score(self.tracker_cfg.final_score_policy) > self.tracker_cfg.track_min_score for track in video_tracks]
            video_tracks = [track for i, track in enumerate(video_tracks) if keep[i]]

        # Sanity check all masks are encoded
        if not self.hungarian_matcher.use_binary_mask_iou:
            for track in video_tracks:
                track.encode_all_masks()

        if args.out_viz_path:
            for track in video_tracks:
                track.process_centroid(video.original_size)
            if args.merge_tracks:
                visualize_results_merged(video.images_folder, video.video_clips, video_tracks, self.tracker_cfg.final_class_policy,
                                           self.tracker_cfg.final_score_policy, out_path=args.out_viz_path, class_name=cat_names)
            else:
                visualize_tracks_independently(video.images_folder, video.video_clips, video_tracks, self.tracker_cfg.final_class_policy,
                                           self.tracker_cfg.final_score_policy, out_path=args.out_viz_path, class_name=cat_names)

        final_tracks_result = [track.get_formatted_result(video.video_id, self.tracker_cfg.final_class_policy, self.tracker_cfg.final_score_policy) for track in video_tracks]

        return final_tracks_result, all_times


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
    all_times = []
    # video_names = ["1f17cd7c","3d04522a", "4d6a99ec", "9ed568e9", "012b09a0", "7223bf62", "d50fa72e", "caf53839"]
    # video_names = ["4348676053", "24947a9f29", "33e8066265", "4b1a561480", "2e21c7e59b", "0e4068b53f"]
    # video_names = ["4348676053",]
    # video_names = ["0e4068b53f"]

    # video_names = ["c8518f8355", "54ad024bb3"]
    # video_names = ["0b97736357"]

    # video_names = ["4e94ce2905"]
    # video_names = ["1957557fa6", "4b1a561480"]
    # video_names = ["0fc3e9dbcc",]
    # video_names = ["c34989e3", "ba5644c3", "2112a80d",]
    # video_names = ["3abe72f8ad"]

    # video_names = ["98d4c5908c", "ad3ff1fb2e", "bfb346a33b", "c35af243fd"]
    # video_names = ["d19a9adf68", "bfb346a33b",]
    # video_names = ["eb49ce8027"]

    for idx, video in tqdm.tqdm(enumerate(data_loader_val)):
        if video.video_clips[0][0].split("/")[0] not in video_names:
            continue
        video_tracks, all_times = tracker(video, device, all_times, args)
        all_tracks.extend(video_tracks)

    finish_time = time.time()
    print(f"Total time non-upsampling {sum(all_times)}")
    print(f"FPS total num frames {8289 / sum(all_times)}")


    print(f"Total time {finish_time-init_time}")
    print(f" Max memory allocated {int(torch.cuda.max_memory_allocated() / MB)}")

    gathered_preds = utils.all_gather(all_tracks)
    results_accums = utils.accumulate_results(gathered_preds)

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
    if not hasattr(model_args, "with_class_inst_attn"):
        model_args.with_class_inst_attn = False
    if not hasattr(model_args, "softmax_activation"):
        model_args.softmax_activation = False
    if not hasattr(model_args, "with_class_no_obj"):
        model_args.with_class_no_obj = False
    if not hasattr(model_args, "use_ct_distance"):
        model_args.use_ct_distance = False
    if not hasattr(model_args, "enc_connect_all_embeddings"):
        model_args.enc_connect_all_embeddings = False
    if not hasattr(model_args, "dec_sort_temporal_offsets"):
        model_args.dec_sort_temporal_offsets = False
    if not hasattr(model_args, "non_temporal_decoder"):
        model_args.non_temporal_decoder = False
    if not hasattr(model_args, "enc_use_new_sampling_init_default"):
        model_args.enc_use_new_sampling_init_default = False
    if not hasattr(model_args, "new_temporal_connection"):
        model_args.new_temporal_connection = False
    if not hasattr(model_args, "viz_att_maps"):
        model_args.viz_att_maps = False



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