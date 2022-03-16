"""
Training script of VisTR
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from util.weights_loading_utils import adapt_weights_evis, adapt_weights_vistr, adapt_weights_unified_model
from util.visdom_vis import build_visualizers, get_vis_win_names

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--debug', action='store_true', help="Use DeformableVisTR")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_extra_class', action='store_true', help="Use DeformableVisTR")
    parser.add_argument('--no_load_class_neurons', action='store_false', help="Use DeformableVisTR")
    parser.add_argument('--finetune_query_embds', action='store_true', help="Use DeformableVisTR")
    parser.add_argument('--finetune_attn_mask_head', action='store_true', help="Use DeformableVisTR")
    parser.add_argument('--finetune_temporal_modules', action='store_true', help="Use DeformableVisTR")

    parser.add_argument('--use_specific_lrs', action='store_true', help="Use DeformableVisTR")
    parser.add_argument('--lambda_lr', action='store_true', help="Use DeformableVisTR")
    parser.add_argument('--lr_new_modules_names', default=["mask_head.lay1_2","mask_head.gn1_2", "mask_head.out_lay1",
                                                           "mask_head.out_gn1", "mask_head.out_lay2", "temporal_attention_weights", "temporal_embd", "insmask_head", "class_embed"], type=str, nargs='+')
    parser.add_argument('--lr_new_modules', default=2, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_curr_proj_names', default=['self_attn.sampling_offsets', 'cross_attn.sampling_offsets', 'reference_points'], type=str, nargs='+')
    parser.add_argument('--lr_linear_curr_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_linear_temp_proj_names', default=['temporal_sampling_offsets',], type=str, nargs='+')
    parser.add_argument('--lr_linear_temp_proj_mult', default=0.1, type=float)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=18, type=int)
    parser.add_argument('--lr_drop', default=[8, 12], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--softmax_activation', action='store_true', help="Use DeformableVisTR")
    # Model parameters
    parser.add_argument('--deformable', action='store_true',
                        help="Use DeformableVisTR")
    parser.add_argument('--freeze_vistr', action='store_true',
                        help="Use DeformableVisTR")
    parser.add_argument('--resume_shift_neuron', action='store_true',
                        help="Use DeformableVisTR")
    # * Training
    parser.add_argument('--eval_only', action='store_true',
                        help="Use DeformableVisTR")
    parser.add_argument('--resume_optim', action='store_true',
                        help="Use DeformableVisTR")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str,
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads_enc', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--nheads_dec', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--num_queries', default=360, type=int, help="Number of query slots")
    parser.add_argument('--use_trajectory_queries', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--force_bug', action='store_true',)


    # * Deformable DeTR
    parser.add_argument('--non_temporal_encoder', action='store_true', help="Enables focal loss ")
    parser.add_argument('--non_temporal_decoder', action='store_true', help="Enables focal loss ")
    parser.add_argument('--new_temporal_connection', action='store_true', help="Enables focal loss")

    parser.add_argument('--num_feature_levels', default=4, type=int, help="Number of query slots")
    parser.add_argument('--with_class_inst_attn',  help="Enables focal loss ",  action='store_true')
    parser.add_argument('--with_ref_point_refine', dest='with_ref_point_refine', action='store_true', help="Enables focal loss ")
    parser.add_argument('--with_box_refine', dest='with_box_refine', action='store_true', help="Enables focal loss ")
    parser.add_argument('--with_gradient', action='store_true', help="Enables focal loss ")
    parser.add_argument('--with_single_class_embed', action='store_true', help="Enables focal loss ")

    parser.add_argument('--with_class_no_obj', action='store_true',
                        help="Enables focal loss ")

    parser.add_argument('--with_loss_vistr_policy', action='store_true',
                        help="Enables focal loss ")

    parser.add_argument('--with_decoder_instance_self_attn', action='store_true',
                        help="Enables focal loss ")
    parser.add_argument('--with_decoder_frame_self_attn', action='store_true',
                        help="Enables focal loss ")

    parser.add_argument('--two_stage', dest='two_stage', action='store_true',
                        help="Enables focal loss ")
    parser.add_argument('--with_embedding_correlation', action='store_true',
                        help="Enables focal loss ")
    parser.add_argument('--with_embedding_correlation_alignment', action='store_true',
                        help="Enables focal loss ")
    parser.add_argument('--enc_temporal_window', default=2, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_temporal_window', default=2, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_curr_points', default=4, type=int,
                        help="Number of query slots")
    parser.add_argument('--enc_n_curr_points', default=4, type=int,
                        help="Number of query slots")

    parser.add_argument('--dec_n_temporal_points', default=2, type=int,
                        help="Number of query slots")
    parser.add_argument('--enc_n_temporal_points', default=2, type=int,
                        help="Number of query slots")

    parser.add_argument('--dec_connect_all_embeddings',  action='store_true', help="Number of query slots")
    parser.add_argument('--dec_sort_temporal_offsets',  action='store_true', help="Number of query slots")
    parser.add_argument('--enc_connect_all_embeddings',  action='store_true', help="Number of query slots")
    parser.add_argument('--enc_use_new_sampling_init_default',  action='store_true', help="Number of query slots")


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--new_segm_module', type=str,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_head_out_mdcn', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_attn_alignment', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--use_ct_distance', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--use_box_coords', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--only_positive_matches', action='store_true',
                        help="Compute mask for only positive matches")
    parser.add_argument('--use_deformable_conv', action='store_true',
                        help="Compute mask for only positive matches")
    parser.add_argument('--mask_head_used_features', default=[['/32', 'encoded'], ['/16', 'encoded'], ['/8', 'encoded'], ['/4', 'backbone']],
                        help="Compute mask for only positive matches")
    parser.add_argument('--att_maps_used_res', default=['/32', '/16', '/8'],
                        help="Compute mask for only positive matches")

    # Loss
    parser.add_argument('--balance_aux_loss', dest='balance_aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--balance_mask_aux_loss', dest='balance_mask_aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--focal_loss', dest='focal_loss', action='store_true',
                        help="Enables focal loss ")
    parser.add_argument('--volumetric_mask_loss', action='store_true', help="Enables focal loss ")
    parser.add_argument('--mask_aux_loss', default=[], type=int, nargs='+',
                        help="List containing the levels in which compute aux_mask_loss")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--use_giou', action='store_true')
    parser.add_argument('--use_l1_distance_sum', action='store_true')

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--centroid_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--create_bbx_from_mask', action='store_true')
    parser.add_argument('--use_non_valid_class', action='store_true')
    parser.add_argument('--use_instance_level_classes', action='store_true')
    parser.add_argument('--transform_strategy', default='vistr')
    parser.add_argument('--temporal_coherence', action='store_true')
    parser.add_argument('--reversed_sampling', action='store_true')
    parser.add_argument('--max_size', default=800, type=int)
    parser.add_argument('--out_scale', default=1, type=float)
    parser.add_argument('--val_width', default=300, type=int)

    parser.add_argument('--transform_pipeline', default='default')

    parser.add_argument('--dataset_type', default='vis')
    parser.add_argument('--timer_path', default='/usr/prakt/p028/data')

    parser.add_argument('--data_path', default='/usr/prakt/p028/data')
    parser.add_argument('--train_set', default='train_train_val_split')
    parser.add_argument('--val_set', default='valid_train_val_split')

    parser.add_argument('--output_dir', default='r101_vistr',
                        help='path where to save, empty for no saving')

    parser.add_argument('--save_model_interval', default=4, type=int,
                        help='epoch interval for model saving. if 0 only save last and best models')

    parser.add_argument('--device', default='cuda',  help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    parser.add_argument('--val_interval',   default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #evaluation parameters
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

    parser.add_argument('--overlap_window', default=2, type=int, help='number of distributed processes')

    parser.add_argument('--start_eval_epoch', default=6, type=int)
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--out_viz_path', default="")
    parser.add_argument('--save_raw_detections',  action='store_true')
    parser.add_argument('--save_clip_viz',  action='store_true')
    parser.add_argument('--save_result', default="")
    parser.add_argument('--merge_tracks', action='store_true')


    parser.add_argument('--train_videos_eval',  default=300, type=int)
    parser.add_argument('--val_videos_eval',  default=None, type=int)
    parser.add_argument('--viz_att_maps', action='store_true')

    #Validation launch
    parser.add_argument('--input_folder', default="")
    parser.add_argument('--epochs_to_eval', default=[5,6,7,8,9,10])

    #visdom parameters
    parser.add_argument('--no_vis', action='store_true',)
    parser.add_argument('--resume_vis', action='store_true')
    parser.add_argument('--vis_port', default=8090, type=int, help='url used to set up distributed training')
    parser.add_argument('--vis_and_log_interval', default=100, type=int, help='url used to set up distributed training')
    parser.add_argument('--vis_server', default="http://localhost", help='url used to set up distributed training')

    return parser


def main(args):
    args.is_training = True
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if args.deformable:
        assert args.batch_size == 1, "DefVisTR segmentation implementation only works with batch_size == 1 to benefit from speed increase"

    train_dataset, num_classes = build_dataset(image_set="train", args=args)
    dataset_val, _ = build_dataset(image_set="val", args=args, num_videos_to_eval=args.val_videos_eval)


    model, criterion, postprocessors = build_model(num_classes, args)
    model.to(device)

    visualizers = build_visualizers(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_parameters = sum(p.numel() for p in model.parameters())
    print('Total num params:', n_total_parameters)
    print('Number of training params:', n_parameters)

    utils.print_training_params(model_without_ddp, args)
    if args.use_specific_lrs:
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                     if not utils.match_name_keywords(n,
                                                      args.lr_backbone_names + args.lr_linear_curr_proj_names +
                                                      args.lr_linear_temp_proj_names + args.lr_new_modules_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if
                           utils.match_name_keywords(n, args.lr_new_modules_names) and p.requires_grad],
                "lr": args.lr * args.lr_new_modules,
            },

            {
                "params": [p for n, p in model_without_ddp.named_parameters() if
                           utils.match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if
                           utils.match_name_keywords(n, args.lr_linear_curr_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_curr_proj_mult,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if
                           utils.match_name_keywords(n, args.lr_linear_temp_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_temp_proj_mult,
            }
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        if args.lambda_lr:
            assert len(args.lr_drop) == 2

            # # Specific schedule
            # #lambda1 Fine-tuned parameters excluding backbone and linear_curr (reference_point, current frame sampling offsets)
            # lambda1 = lambda epoch_: 1 if epoch_ < args.lr_drop[1] else 0.1
            #
            # #lambda2 New parameters init from scratch excluding temporal sampling offsets
            # lambda2 = lambda epoch_:  1 if epoch_ < args.lr_drop[0] else (0.1 if epoch_ < args.lr_drop[1] else 0.01)
            #
            # # lambda3 Backbone
            # lambda3 = lambda epoch_: 1 if epoch_ < args.lr_drop[1] else 0.1
            #
            #
            # # lambda4 Fine-tuned current frame sampling offsets and reference point
            # lambda4 = lambda epoch_: 1 if epoch_ < args.lr_drop[1] else 0.1
            #
            # # lambda5 Temporal offsets
            # lambda5 = lambda epoch_:  1 if epoch_ < args.lr_drop[0] else (0.1 if epoch_ < args.lr_drop[1] else 0.01)


            # Specific schedule
            # #lambda1 Fine-tuned parameters excluding backbone and linear_curr (reference_point, current frame sampling offsets)
            # lambda1 = lambda epoch_: 1 if epoch_ < args.lr_drop[0] else (0.1 if epoch_ < args.lr_drop[1] else 0.01)
            #
            # #lambda2 New parameters init from scratch excluding temporal sampling offsets
            # lambda2 = lambda epoch_:  1 if epoch_ < args.lr_drop[0] else (0.05 if epoch_ < args.lr_drop[1] else 0.01)
            #
            # # lambda3 Backbone
            # lambda3 = lambda epoch_: 1 if epoch_ < args.lr_drop[0] else (0.1 if epoch_ < args.lr_drop[1] else 0.01)
            #
            # # lambda4 Fine-tuned current frame sampling offsets and reference point
            # lambda4 = lambda epoch_: 1 if epoch_ < args.lr_drop[0] else (0.1 if epoch_ < args.lr_drop[1] else 0.01)
            #
            # # lambda5 Temporal offsets
            # lambda5 = lambda epoch_:  1 if epoch_ < args.lr_drop[0] else (0.05 if epoch_ < args.lr_drop[1] else 0.01)

            #lambda1 Fine-tuned parameters excluding backbone and linear_curr (reference_point, current frame sampling offsets)
            lambda1 = lambda epoch_: 1 if epoch_ < args.lr_drop[0] else (0.2 if epoch_ < args.lr_drop[1] else 0.02)

            #lambda2 New parameters init from scratch excluding temporal sampling offsets
            lambda2 = lambda epoch_:  1 if epoch_ < args.lr_drop[0] else (0.1 if epoch_ < args.lr_drop[1] else 0.01)

            # lambda3 Backbone
            lambda3 = lambda epoch_: 1 if epoch_ < args.lr_drop[0] else (0.1 if epoch_ < args.lr_drop[1] else 0.01)

            # lambda4 Fine-tuned current frame sampling offsets and reference point
            lambda4 = lambda epoch_: 1 if epoch_ < args.lr_drop[0] else (0.2 if epoch_ < args.lr_drop[1] else 0.02)

            # lambda5 Temporal offsets
            lambda5 = lambda epoch_:  1 if epoch_ < args.lr_drop[0] else (0.1 if epoch_ < args.lr_drop[1] else 0.01)

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2, lambda3, lambda4, lambda5])


        else:
            if len(args.lr_drop) == 1:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop[0])
            else:
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    else:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone.0" not in n and p.requires_grad]},

            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone.0" in n and p.requires_grad],
                "lr": args.lr_backbone},
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers, worker_init_fn=seed_worker)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, collate_fn=utils.val_collate, num_workers=args.num_workers)

    data_loader_val_train = None
    if args.eval_train:
        val_train_dataset, _ = build_dataset(image_set="val", args=args, split=args.train_set, num_videos_to_eval=args.train_videos_eval)
        if args.distributed:
            sampler_val_train = DistributedSampler(val_train_dataset, shuffle=False)
        else:
            sampler_val_train = torch.utils.data.SequentialSampler(val_train_dataset)

        data_loader_val_train =  DataLoader(val_train_dataset, args.batch_size, sampler=sampler_val_train, collate_fn=utils.val_collate)

    output_dir = Path(args.output_dir)

    best_val_stats = None
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_state_dict = model_without_ddp.state_dict()
        if args.deformable:

            if args.new_temporal_connection:
                resume_state_dict = adapt_weights_unified_model(checkpoint, model_state_dict,  args.num_feature_levels, args.focal_loss, args.no_load_class_neurons, args.num_frames,
                                            args.finetune_query_embds, args.use_trajectory_queries, args.enc_n_curr_points, args.dec_n_curr_points)

            else:
                resume_state_dict = adapt_weights_evis(checkpoint, model_state_dict, args.num_feature_levels, args.focal_loss,
                                                   args.no_load_class_neurons, args.num_frames, args.finetune_query_embds,
                                                   args.finetune_attn_mask_head, args.use_trajectory_queries, args.with_decoder_instance_self_attn, args.finetune_temporal_modules,
                                                   args.enc_temporal_window, args.enc_n_temporal_points, args.dec_n_temporal_points)

        else:
            # load coco pretrained weight
            checkpoint = torch.load(args.resume, map_location='cpu')['model']
            # model.module.load_state_dict(checkpoint,strict=False)
            resume_state_dict = adapt_weights_vistr(checkpoint, model_state_dict, args.eval_only)

        model_without_ddp.load_state_dict(resume_state_dict)


        # RESUME OPTIM
        if not args.eval_only and args.resume_optim:
            if 'optimizer' in checkpoint:
                for c_p, p in zip(checkpoint['optimizer']['param_groups'], param_dicts):
                    c_p['lr'] = p['lr']
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
            # if 'best_val_stats' in checkpoint:
            #     best_val_stats = checkpoint['best_val_stats']


        if not args.eval_only and args.resume_vis and 'vis_win_names' in checkpoint:
            for k, v in visualizers.items():
                for k_inner in v.keys():
                    visualizers[k][k_inner].win = checkpoint['vis_win_names'][k][k_inner]


    if args.eval_only:
        al_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, device, output_dir, visualizers['val'], args, args.start_epoch)
        return

    # RESUME VIS


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            visualizers['train'], args)

        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if args.save_model_interval and not epoch % int(args.save_model_interval):
                checkpoint_paths.append(output_dir / f"checkpoint_epoch_{epoch}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'vis_win_names': get_vis_win_names(visualizers),
                    'best_val_stats': best_val_stats
                }, checkpoint_path)

        # # VAL
        checkpoint_paths = []
        if (epoch == 1 or not epoch % args.val_interval) and epoch >= args.start_eval_epoch:
            if not data_loader_val.dataset.has_gt:
                args.save_result = os.path.join("eval_results", f"epoch_{epoch}")

            val_stats = evaluate(
                model, criterion, postprocessors, data_loader_val, device, output_dir, visualizers['val'], args, epoch)

            if val_stats is not None and data_loader_val.dataset.has_gt:
                stat_names = ['TRACK_mAP_IoU_0_50-0_95', 'TRACK_mAR_IoU_0_50-0_95']
                if best_val_stats is None:
                    best_val_stats = val_stats
                best_val_stats = [best_stat if best_stat > stat else stat
                                  for best_stat, stat in zip(best_val_stats, val_stats)]
                for b_s, s, n in zip(best_val_stats, val_stats, stat_names):
                    if b_s == s:
                        checkpoint_paths.append(output_dir / f"checkpoint_best_{n}.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
