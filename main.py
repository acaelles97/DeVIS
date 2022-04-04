"""
Training script of DeVIS
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
import yaml
import random
from contextlib import redirect_stdout
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import evaluate_coco, inference_vis, train_one_epoch
from models import build_model, build_tracker
from util.weights_loading_utils import adapt_weights_devis
from util.visdom_vis import build_visualizers, get_vis_win_names
from config import get_cfg_defaults


def get_args_parser():
    parser = argparse.ArgumentParser('DeVIS argument parser', add_help=False)

    parser.add_argument('--config-file',  help="Run test only")

    parser.add_argument('--eval-only', action='store_true', help="Run test only")

    parser.add_argument('--seed', default=42, type=int)

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def sanity_check(cfg):
    print("a")
    assert isinstance(cfg.mask_aux_loss, list), "args.mask_aux_loss not a list"
    assert len(cfg.mask_aux_loss) == len(set(cfg.mask_aux_loss)), f"Use unique levels number in args.mask_aux_loss, Value {cfg.mask_aux_loss}"
    assert min(cfg.mask_aux_loss) >= 0 and max(cfg.mask_aux_loss) <= 4, f"Available aux_loss levels : [0, 1, 2, 3, 4], Value {cfg.mask_aux_loss}"
    # AUX_LOSS_WEIGHTING_COEF and num_layers != 6
    # if bbx refine no ref point refine.
    # 'num_out': cfg.TEST.NUM_OUT != num_queries if not USE_TOP_K
    # batch size 1 if dataset type vis
    # calcular stride i >= 1


# def fix_seed(cfg):
#     # fix the seed for reproducibility
#     seed = cfg.SEED + utils.get_rank()
#
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def main(args, cfg):
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()

    print(f"seed {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # def seed_worker(worker_id):
    #     np.random.seed(np.random.get_state()[1][0] + worker_id)
    #     # worker_seed = torch.initial_seed() % 2 ** 32
    #     # print(f"worker_id {worker_id}")
    #     # print(f"torch.initial_seed() {torch.initial_seed()}")
    #     # np.random.seed(worker_seed)
    #     # random.seed(worker_seed)
    #     print(f"worker_seed {np.random.get_state()[1][0] + worker_id}")

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # print(f"worker seed {worker_seed}")

    g = torch.Generator()
    g.manual_seed(0)


    train_dataset, num_classes = build_dataset(image_set="TRAIN", cfg=cfg)
    dataset_val, _ = build_dataset(image_set="VAL", cfg=cfg)


    # print(f"Torch random {torch.randint(0, 300, size=(1, )).item()}")
    # print(f"Torch random {torch.randint(0, 300, size=(1, )).item()}")
    # print(f"Torch random {torch.randint(0, 300, size=(1, )).item()}")
    model, criterion, postprocessors = build_model(num_classes, device, cfg)
    model.to(device)

    visualizers = build_visualizers(cfg)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    tracker = None
    if cfg.DATASETS.TYPE == 'vis':
        tracker = build_tracker(model, cfg)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    print(f'Total num params: {n_total_params}')
    print(f'Number of training params: {n_train_params}')

    utils.print_training_params(model_without_ddp, cfg)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not utils.match_name_keywords(n,
                                                  cfg.SOLVER.BACKBONE_NAMES + cfg.SOLVER.LR_LINEAR_PROJ_NAMES +
                                                  cfg.SOLVER.LR_MASK_HEAD_NAMES + cfg.SOLVER.DEVIS.LR_TEMPORAL_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR,
        },

        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       utils.match_name_keywords(n, cfg.SOLVER.BACKBONE_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.LR_BACKBONE,
        },

        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       utils.match_name_keywords(n, cfg.SOLVER.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_LINEAR_PROJ_MULT,
        },

        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       utils.match_name_keywords(n, cfg.SOLVER.LR_MASK_HEAD_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_MASK_HEAD_MULT,
        },

        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       utils.match_name_keywords(n, cfg.SOLVER.DEVIS.LR_TEMPORAL_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DEVIS.LR_TEMPORAL_LINEAR_PROJ_MULT,
        }

    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS)

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.SOLVER.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS, worker_init_fn=seed_worker, generator=g)

    data_loader_val = DataLoader(dataset_val, cfg.SOLVER.BATCH_SIZE, sampler=sampler_val, collate_fn=utils.val_collate, num_workers=cfg.NUM_WORKERS)

    output_dir = Path(cfg.OUTPUT_DIR)

    best_val_stats = None
    if cfg.MODEL.WEIGHTS:
        if cfg.MODEL.WEIGHTS.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.MODEL.WEIGHTS, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')

        # resume_state_dict = {}
        # checkpoint = torch.load("/usr/stud/cad/results/trainings/debug/out_to_check/weights_old.pth")
        # for key, value in checkpoint.items():
        #     if key == 'vistr.backbone.1.temporal_embd':
        #         resume_state_dict['def_detr.backbone.1.temporal_embed'] = value
        #     elif key.startswith("vistr"):
        #         resume_state_dict[key.replace('vistr', 'def_detr')] = value
        #     else:
        #         resume_state_dict[key] = value

        model_state_dict = model_without_ddp.state_dict()
        if cfg.DATASETS.TYPE == 'vis':
            if args.eval_only:
                resume_state_dict = {}
                for key, value in checkpoint['model'].items():
                    if key == 'vistr.backbone.1.temporal_embd':
                        resume_state_dict['def_detr.backbone.1.temporal_embed'] = value
                    elif key.startswith("vistr"):
                        resume_state_dict[key.replace('vistr', 'def_detr')] = value
                    else:
                        resume_state_dict[key] = value

            else:
                resume_state_dict = adapt_weights_devis(checkpoint, model_state_dict, cfg.MODEL.NUM_FEATURE_LEVELS, cfg.MODEL.LOSS.FOCAL_LOSS,
                                                        cfg.SOLVER.DEVIS.FINETUNE_CLASS_LOGITS, cfg.MODEL.DEVIS.NUM_FRAMES, cfg.SOLVER.DEVIS.FINETUNE_QUERY_EMBEDDINGS,
                                                        cfg.MODEL.DEVIS.INSTANCE_LEVEL_QUERIES, cfg.SOLVER.DEVIS.FINETUNE_TEMPORAL_MODULES,
                                                        cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_CONNECT_ALL_FRAMES,
                                                        cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_TEMPORAL_WINDOW, cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_N_POINTS_TEMPORAL_FRAME,
                                                        cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.DEC_N_POINTS_TEMPORAL_FRAME)



        model_without_ddp.load_state_dict(resume_state_dict, strict=True)

        # RESUME OPTIM
        if not args.eval_only and cfg.SOLVER.RESUME_OPTIMIZER:
            if 'optimizer' in checkpoint:
                for c_p, p in zip(checkpoint['optimizer']['param_groups'], param_dicts):
                    c_p['lr'] = p['lr']
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                cfg.START_EPOCH = checkpoint['epoch'] + 1
            if 'best_val_stats' in checkpoint:
                best_val_stats = checkpoint['best_val_stats']

        if not args.eval_only and cfg.RESUME_VIS and 'vis_win_names' in checkpoint:
            for k, v in visualizers.items():
                for k_inner in v.keys():
                    visualizers[k][k_inner].win = checkpoint['vis_win_names'][k][k_inner]

    if args.eval_only:
        if cfg.DATASETS.TYPE == 'vis':
            _ = inference_vis(
                tracker, data_loader_val, dataset_val, visualizers['val'], device, output_dir, cfg.TEST.SAVE_PATH, 0)

        else:
            _, coco_evaluator = evaluate_coco(
                model, criterion, postprocessors, data_loader_val, device, output_dir, visualizers['val'], cfg.VISDOM_AND_LOG_INTERVAL, cfg.START_EPOCH)
            if args.OUTPUT_DIR:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(cfg.START_EPOCH, cfg.SOLVER.EPOCHS + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, visualizers['train'], cfg.VISDOM_AND_LOG_INTERVAL, cfg.SOLVER.GRAD_CLIP_MAX_NORM)

        lr_scheduler.step()

        checkpoint_paths = [output_dir / 'checkpoint.pth']

        if cfg.SOLVER.CHECKPOINT_INTERVAL and not epoch % int(cfg.SOLVER.CHECKPOINT_INTERVAL):
            checkpoint_paths.append(output_dir / f"checkpoint_epoch_{epoch}.pth")

        # # VAL
        if (epoch == 1 or not epoch % cfg.TEST.EVAL_PERIOD) and epoch >= cfg.TEST.START_EVAL_EPOCH:

            if cfg.DATASETS.TYPE == 'vis':
                out_folder_name = os.path.join(cfg.TEST.SAVE_PATH, f"epoch_{epoch}")
                _ = inference_vis(
                    tracker, data_loader_val, dataset_val, visualizers['val'], device, output_dir, out_folder_name, epoch)
                # TODO: If val_dataset has_gt save additionally best epoch

            else:
                val_stats, _ = evaluate_coco(
                    model, criterion, postprocessors, data_loader_val, device,
                    output_dir, visualizers['val'], cfg.VISDOM_AND_LOG_INTERVAL, epoch)

                stat_names = ['BBOX_AP_IoU_0_50-0_95', ]
                if cfg.MODEL.MASK_ON:
                    stat_names.extend(['MASK_AP_IoU_0_50-0_95', ])

                if best_val_stats is None:
                    best_val_stats = val_stats
                best_val_stats = [best_stat if best_stat > stat else stat
                                  for best_stat, stat in zip(best_val_stats, val_stats)]

                for b_s, s, n in zip(best_val_stats, val_stats, stat_names):
                    if b_s == s:
                        checkpoint_paths.append(output_dir / f"checkpoint_best_{n}.pth")

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': cfg,
                'vis_win_names': get_vis_win_names(visualizers),
                'best_val_stats': best_val_stats
            }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeVIS training and evaluation script', parents=[get_args_parser()])
    args_ = parser.parse_args()

    cfg_ = get_cfg_defaults()
    cfg_.merge_from_file(args_.config_file)
    cfg_.merge_from_list(args_.opts)
    cfg_.freeze()
    if cfg_.OUTPUT_DIR:
        Path(cfg_.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg_.OUTPUT_DIR, 'config.yaml'), 'w') as yaml_file:
            with redirect_stdout(yaml_file):
                print(cfg_.dump())

    main(args_, cfg_)
