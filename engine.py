"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
import os
import sys
from typing import Iterable
import pickle
import torch
import util.misc as utils
from timeit import default_timer as timer
from inference_2 import run_inference
from pathlib import Path

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, visualizers, args):

    vis_iter_metrics = None
    if visualizers:
        vis_iter_metrics = visualizers['iter_metrics']

    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=vis_iter_metrics,
        debug=False)
    metric_logger.add_meter('lr_finetuned_params', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_new_params', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_backbone', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_curr_sampling', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_temporal_sampling', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))


    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch)):
    #     if i >= 5:
    #         break
    # for i, (samples, targets) in enumerate(data_loader):
    #     if True:
    #         continue
    #     if targets[0]["valid"].shape[0] <= 6 or torch.sum(targets[0]["valid"]) == 6:
    #         continue
    #     if all(targets[0]["original_valid"]):
    #         continue
    #     if not targets[0]["original_valid"].sum() == 0:
    #         continue
    #     if i < 5:
    #         continue
    #         break
    #     print(f"Target id {targets[0]['image_id']}")
        # if targets[0]['image_id'] != 21809:
        #     continue
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # if torch.all(targets[0]["valid"]):
        #     continue
        # if (targets[0]["valid"].shape[0] <= args.num_frames):
        #     continue
        outputs = model(samples, targets)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # print(f"Loss value {loss_value}")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()

        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr_finetuned_params=optimizer.param_groups[0]["lr"],
                             lr_new_params=optimizer.param_groups[1]["lr"],
                             lr_backbone=optimizer.param_groups[2]["lr"],
                             lr_curr_sampling=optimizer.param_groups[3]["lr"],
                             lr_temporal_sampling=optimizer.param_groups[4]["lr"]

    )


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader,  device, output_dir, visualizers, args, epoch):
    model.eval()
    class_av_ap_all, class_av_ar_all = run_inference(model, data_loader, device, args)
    eval_stats =  [class_av_ap_all, class_av_ar_all]

        # VIS
    if visualizers and class_av_ap_all is not None:
        visualizers['epoch_eval'].plot(eval_stats, epoch)

    return eval_stats