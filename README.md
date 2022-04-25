# DeVIS: Making Deformable Transformers Work for Video Instance Segmentation

This repository provides the official implementation of the [DeVIS: Making Deformable Transformers Work for Video Instance Segmentation](https://arxiv.org/abs/2101.02702) paper by Adria Caelles, [Tim Meinhardt](https://dvl.in.tum.de/team/meinhardt/), [Guillem Brasó](https://dvl.in.tum.de/team/braso/) and  [Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/). The codebase builds upon [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [VisTR](https://github.com/Epiphqny/VisTR) and [TrackFormer](https://github.com/timmeinhardt/trackformer).

<!-- **As the paper is still under submission this repository will continuously be updated and might at times not reflect the current state of the [arXiv paper](https://arxiv.org/abs/2012.01866).** -->

<div align="center">
    <img src="docs/evis_method.png" width="800"/>
</div>

## Abstract
Video Instance Segmentation (VIS) jointly tackles multi-object detection, tracking, and segmentation in video sequences. 
In the past, VIS methods mirrored the fragmentation of these subtasks in their architectural design, hence missing out on a joint solution. 
Transformers recently allowed to cast the entire VIS task as a single set-prediction problem. Nevertheless, the quadratic complexity of existing Transformer-based VIS methods requires long training times, high memory requirements, and processing of low-single-scale feature maps.
Deformable attention provides a more efficient alternative but its application to the temporal domain or the segmentation task have not yet been explored.
In this work, we present Deformable VIS (DeVIS), a VIS method which capitalizes on the efficiency and performance of deformable Transformers. 
To reason about all VIS subtasks jointly over multiple frames, we present temporal multi-scale deformable attention with instance-aware object queries.
We further introduce a new image and video instance mask head which exploits multi-scale features, and perform near-online video processing with multi-cue clip tracking.
DeVIS benefits from comparatively small memory as well as training time requirements, and achieves state-of-the-art results on the YouTube-VIS 2019 and 2021, as well as the challenging OVIS dataset.


#Model zoo
Deformable Mask Head

| Model | AP    | AP50 | AP75 | AR1 | AR10 | Pretrain |
|-------|-------|------|------|-----|------|----------|
| dadad | dada  |      |      |     |      |          |

YT-19 

YT-21

OVIS

#Usage

## Features from this repo
* Multi-GPU test
* Evaluation during training
* Training and val curves visualization using visdom
* Visualize results and attention maps! (*Only available for DeVIS*)

## Installation

1. Clone and enter this repository:
    ```
    git clone git@github.com:acaelles97/DeVIS.git
    cd DeVIS
    ```
2. Install packages for Python 3.8:
   1. Install PyTorch 1.11.0 and torchvision 0.12.0 from [here](https://pytorch.org/get-started/locally/).  
   2. `pip3 install -r requirements.txt`
   3. Install [youtube-vis](https://github.com/youtubevos/cocoapi) api
   4. Install MultiScaleDeformableAttention package: `python src/models/ops/setup.py build_ext install`

## Configuration 
We have been inspired by [detectron2](https://github.com/facebookresearch/detectron2) in order to build our configuration system. 
We hope this allows the research community to more easily build upon our method.
Refer to `config.py` to get an overview of all the configuration options available including how the model is built, training and test options.

## Dataset preparation
We expect the following organization for COCO, YT-19, YT-21 & OVIS training. 
User must set cfg.DATASETS.DATA_PATH to the root data path. 
We refer to `src/datasets/coco.py` & `src/datasets/vis.py` to modify the expected format for COCO dataset and VIS datasets respectively.

```
cfg.DATASETS.DATA_PATH/
└── COCO/
  ├── train2017/
  ├── val2017/
  └── annotations/
      ├── instances_train2017.json
      └── instances_val2017.json
 
└── Youtube_VIS/
  ├── train/
      ├── JPEGImages
      └── train.json 
  └── valid/
      ├── JPEGImages
      └── valid.json 

└── Youtube_VIS-2021/
  ├── train/
      ├── JPEGImages
      └── instances.json 
  └── valid/
      ├── JPEGImages
      └── instances.json

└── OVIS/
  ├── train/
  ├── annotations_train.json/
  ├── valid/     
  └── annotations_valid.json/

```

## Visdom
Monitoring of the training/evaluation progress is possible via command line as well as [Visdom](https://github.com/fossasia/visdom). 
For the latter, a Visdom server must be running at `VISDOM_PORT=8090` and `VISDOM_SERVER=http://localhost`. 
To deactivate Visdom logging set `VISDOM_ON=False`.

## Train
We provide configurations files to train Deformable DeTR `configs/deformable_detr/deformable_detr_R_50.yaml`, Deformable Mask-Head `configs/deformable_detr/deformable_mask_head_R_50.yaml` and DeVIS `configs/devis/devis_R_50.yaml`.
For instance, the command to train DeVIS on YT-21 with 4GPUs is as following:
```
torchrun --nproc_per_node=4 main.py --config-file configs/devis/devis_R_50.yaml DATASETS.TRAIN_DATASET yt_vis_train_21 DATASETS.VAL_DATASET yt_vis_val_21
```
As shown above, user can override config-file parameters by passing the new KEY VALUE pair.    



## Evaluate
To evaluate model's performance, you just need to add the --eval-only argument and set MODEL.WEIGHTS to the checkpoint path.
For example, the following command shows 
```
torchrun --nproc_per_node=4 main.py --config-file configs/devis/devis_R_50.yaml --eval-only DATASETS.VAL_DATASET yt_vis_val_21
```
### Visualize results
We also provide configuration options to save results for visualization (only for DeVIS). 
When `TEST.VIZ.OUT_VIZ_PATH=path/to/save` is specified, the visual results from the .json file will be saved.
Additionally, `TEST.VIZ.SAVE_CLIP_VIZ` allows saving results from the sub-clips too, being useful to get an idea of the model performance without the clip stitching being involved.
Finally, `TEST.VIZ.SAVE_MERGED_TRACKS=True` plots all tracks results on the same image (same as figures from the paper). This option requires to properly set `TEST.CLIP_TRACKING.MIN_TRACK_SCORE`, `TEST.CLIP_TRACKING.MIN_FRAME_SCORE` and `TEST.NUM_OUT` for each specific video almost in order to get satisfactory visualization.
We provide an aditional config file  that changes threshold to get more visual appealing results.

```
torchrun --nproc_per_node=4 main.py --config-file configs/devis/devis_R_50_visualization.yaml --eval-only DATASETS.VAL_DATASET yt_vis_val_21 MODEL.WEIGHTS /path/to/checkpoint_file
```

### Attention maps
We also provide an additional script `visualize_att_maps.py` to save attention maps. It does not involve clip stitching, so only allows visualizing results from each video's sub-clips. Visualization as well as `TEST.CLIP_TRACKING.MIN_TRACK_SCORE`, `TEST.CLIP_TRACKING.MIN_FRAME_SCORE` and `TEST.NUM_OUT` are taken into consideration.
```
torchrun --nproc_per_node=1 visualize_att_maps.py --config-file configs/devis/devis_R_50_visualization.yaml DATASETS.VAL_DATASET yt_vis_val_21  MODEL.WEIGHTS /path/to/checkpoint_file
```