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
import tqdm
from zipfile import ZipFile
from itertools import product
from inference_devis import Track, get_args_parser, evaluate_ovis_accums, encode_mask
from models.matcher import HungarianInferenceMatcher



def args_parser():
    parser = argparse.ArgumentParser('VisTR inference script', parents=[get_args_parser()])
    parser.add_argument("--load_results", default="", type=str)
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--save_results", default="", type=str)


    return parser

class Tracker:

    def __init__(self, model, args):
        self.model = model
        self.hungarian_matcher = HungarianInferenceMatcher(cost_mask_iou=args.cost_mask_iou, cost_class=args.cost_class,
                                                           stride=args.overlap_window, score_cost=args.cost_score, cost_center_distance=args.center_distance_cost,
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
                processed_masks.append(mask)
                # if idx == 0:
                #     if t < num_masks - self.overlap_window:
                #         processed_masks.append(encode_mask(mask))
                #     else:
                #         processed_masks.append(mask)
                # else:
                #     if self.overlap_window + start_idx <= t < num_masks - self.overlap_window or t < start_idx:
                #         processed_masks.append(encode_mask(mask))
                #     else:
                #         processed_masks.append(mask)

        return processed_masks


    def update_params(self, new_cfg):
        tracker_cfg = {
            "track_min_detection_score": new_cfg["track_min_detection_score"],
            "track_min_score": new_cfg["track_min_score"],
            "track_min_detections": new_cfg["track_min_detections"],
            "final_class_policy": new_cfg["final_class_policy"],
            "final_score_policy": new_cfg["final_score_policy"],
        }
        hungarian_cfg = {
            "cost_mask_iou": new_cfg["cost_mask_iou"],
            "cost_class": new_cfg["cost_class"],
            "score_cost": new_cfg["score_cost"],
            "cost_center_distance": new_cfg["cost_center_distance"],
            "t_window": new_cfg["t_window"],
            "use_binary_mask_iou": new_cfg["use_binary_mask_iou"],
            "use_frame_average_iou": new_cfg["use_frame_average_iou"],
            "use_center_distance": new_cfg["use_center_distance"],
        }

        self.tracker_cfg = utils.nested_dict_to_namespace(tracker_cfg)
        self.hungarian_matcher = HungarianInferenceMatcher(**hungarian_cfg)

    @torch.no_grad()
    def infer_and_save_results(self, video, device, args):
        sampler_val = torch.utils.data.SequentialSampler(video)
        video_loader = DataLoader(video, 1, sampler=sampler_val, num_workers=args.num_workers)
        real_video_length = video.real_video_length
        clip_length = self.num_frames if real_video_length is None or real_video_length >= self.num_frames else real_video_length
        video_info = {
            "tgt_size": video.original_size,
            "clip_length": clip_length
        }
        video_results = []
        for idx, video_clip in enumerate(video_loader):
            video_clip = video_clip.to(device)
            results = self.model(video_clip.squeeze(0), video_info)
            results["masks"] = results["masks"].to("cpu")
            video_results.append(results)

        return video_results

    @torch.no_grad()
    def match_results(self, video, loaded_results, device, args):
        sampler_val = torch.utils.data.SequentialSampler(video)
        video_loader = DataLoader(video, 1, sampler=sampler_val, num_workers=args.num_workers)
        real_video_length = video.real_video_length
        clip_length = self.num_frames if real_video_length is None or real_video_length >= self.num_frames else real_video_length

        video_tracks = None
        for idx, results in enumerate(loaded_results):
            pred_scores, pred_classes, pred_boxes, pred_masks, pred_centroid_points = results["scores"], results["labels"], results["boxes"], \
                                                                                      results["masks"], results["centroid_points"]
            pred_masks = pred_masks.to(device)
            detected_instances = pred_scores.shape[1]

            start_idx = 0 if idx != len(video_loader) - 1 else video.last_real_idx
            clip_tracks = [Track(track_id, clip_length, start_idx) for track_id in range(detected_instances)]

            # Processing all the masks at the same time was giving problems
            processed_masks_dict = {}
            processed_centroids_dict = {}
            encode_all_masks = self.hungarian_matcher.use_binary_mask_iou
            for i, track in enumerate(clip_tracks):
                mask_id = results['inverse_idxs'][i].item()
                if mask_id not in processed_masks_dict.keys():
                    processed_masks_dict[mask_id] = self.process_masks(start_idx, idx, video.original_size, encode_all_masks,
                                                                       pred_masks[:, mask_id])
                    processed_centroids_dict[mask_id] = pred_centroid_points[:, mask_id]

                track.update(pred_scores[:, i], pred_classes[:, i], pred_boxes[:, i], processed_masks_dict[mask_id],
                             processed_centroids_dict[mask_id], mask_id)

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
                    track.update_stride_and_encode_masks(clip_length - self.overlap_window, self.overlap_window)

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
        final_tracks_result = [
            track.get_formatted_result(video.video_id, self.tracker_cfg.final_class_policy, self.tracker_cfg.final_score_policy) for track in
            video_tracks]

        return final_tracks_result

def run_param_grid_search(data_loader_val,  args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    args.tracker_cfg = None
    args.save_result = ""
    tracker = Tracker(None, args)

    tracker_param_grids = {
        "cost_mask_iou":[2, 3, 4],
        "cost_class":[1, 2],
        "score_cost": [1, 2],
        "cost_center_distance":[0],
        "t_window": [args.overlap_window],
        "use_binary_mask_iou":[False],
        "use_frame_average_iou":[False],
        "use_center_distance":[False],
        "track_min_detection_score": [0.001],
        "track_min_score": [0.004],
        "track_min_detections": [1],
        "final_class_policy": ["most_common"],
        "final_score_policy": ["mean"],
    }

    tracker_cfgs = [dict(zip(tracker_param_grids, v))
                          for v in product(*tracker_param_grids.values())]

    tracker_param_cfgs = []
    for cfg in tracker_cfgs:
        tracker_param_cfgs.append({
            "config": cfg
        })

    loaded_results =  torch.load(args.load_results, map_location='cpu')

    total_num_experiments = len(tracker_param_cfgs)
    print(f'NUM experiments: {total_num_experiments}')

    keys = tracker_param_grids.keys()
    column_values = ",".join(["Name"] + list(keys))
    print(column_values)

    # run all tracker config combinations for all experiment configurations
    exp_counter = 1
    for tracker_cfg in tracker_param_cfgs:
        cfg = tracker_cfg["config"]
        final_tracks = []
        tracker.update_params(cfg)
        for idx, video in enumerate(data_loader_val):
            video_results = loaded_results[video.video_id]
            video_tracks = tracker.match_results(video, video_results, device, args)
            final_tracks.extend(video_tracks)
        if data_loader_val.dataset.has_gt:
            out_eval_folder = os.path.join(args.output_dir, "output_eval")
            if not os.path.exists(out_eval_folder):
                os.makedirs(out_eval_folder, exist_ok=True)
            class_av_ap_all, class_av_ar_all = evaluate_ovis_accums(final_tracks, data_loader_val.dataset.annotations, out_eval_folder)

            tracker_cfg['track_mAP'] = class_av_ap_all
            tracker_cfg['num_detections'] = sum([1 if det is not None else 0 for track in final_tracks for det in track["segmentations"]])

        else:
            out_dir = os.path.join(args.output_dir, args.save_result, f"results_exp{exp_counter:03d}")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"results.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(final_tracks, f)

            out_zip_filename = os.path.join(out_dir, f"results.zip")
            with ZipFile(out_zip_filename, 'w') as zip_obj:
                zip_obj.write(out_file, os.path.basename(out_file))
            os.remove(out_file)

            experiment_info = ",".join([f"results_exp{exp_counter:03d}.zip"] + [str(cfg[key]) for key in keys])
            print(experiment_info)

        exp_counter += 1

    if data_loader_val.dataset.has_gt:
        sorted_idxs = np.array([cfg['track_mAP'] for cfg in tracker_param_cfgs]).argsort()
        for idx in sorted_idxs:
            print(f"Track mAP: {tracker_param_cfgs[idx]['track_mAP']:.4f} Num_detections: {tracker_param_cfgs[idx]['num_detections']:06d} Config: {tracker_param_cfgs[idx]['config']}")

    return

def run_inference(num_classes, model_args, state_dict, data_loader_val, args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tracker_cfg = {
        "track_min_detection_score": args.track_min_detection_score,
        "track_min_score": args.track_min_score,
        "track_min_detections": args.track_min_detections,
        "final_class_policy": args.final_class_policy,
        "final_score_policy": args.final_score_policy,
    }

    args.tracker_cfg = utils.nested_dict_to_namespace(tracker_cfg)

    model, criterion, postprocessors = build_model(num_classes, model_args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    model_without_ddp.load_state_dict(state_dict['model'])
    model.eval()

    tracker = Tracker(model, args)
    results = {}
    for idx, video in tqdm.tqdm(enumerate(data_loader_val)):
        video_tracks = tracker.infer_and_save_results(video, device, args)
        results[video.video_id] = video_tracks

    torch.save(results, args.save_results)

    return

def main(args):
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.input_folder:
        args.resume = os.path.join(args.input_folder, f"checkpoint_epoch_{args.epochs_to_eval[0]}.pth")
        args.save_result = f"val_epoch_{args.epochs_to_eval[0]}"

    state_dict = torch.load(args.resume, map_location="cpu")
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
        model_args.instance_level_queries = False
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

    if not args.load_results:
        assert args.save_results
        print("args.save_results")
        if os.path.exists(args.save_results):
            raise
        run_inference(num_classes, model_args, state_dict, data_loader_val, args)

    else:
        run_param_grid_search(data_loader_val, args)


if __name__ == '__main__':
    parse = args_parser()
    args = parse.parse_args()

    main(args)