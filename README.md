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


#Results
##COCO

| Model                                                        | AP   | AP50 | AP75 | AR1 | AR10 | FPS  |
|--------------------------------------------------------------|------|------|------|-----|------|------|
| [Mask R-CNN](https://github.com/facebookresearch/detectron2) | 37.2 | 58.5 | 39.8 |     |      | 21.4 |
| Ours                                                         | 38.0 | 61.4 | 40.1 |     |      | 12.1 |

##YT-19 

| Model                                             | AP   | AP50 | AP75 | AR1  | AR10 | Pretrain |
|---------------------------------------------------|------|------|------|------|------|----------|
| [VisTR](https://github.com/Epiphqny/VisTR)        |      |      |      |      |      |          |
| [IFC](https://github.com/sukjunhwang/IFC)         |      |      |      |      |      |          |
| [SeqFormer](https://github.com/wjf5203/SeqFormer) |      |      |      |      |      |          |
| Ours                                              | 44.4 | 67.9 | 48.6 | 42.4 | 51.6 |          |

##YT-21

| Model                                             | AP   | AP50 | AP75 | AR1  | AR10 | Pretrain |
|---------------------------------------------------|------|------|------|------|------|----------|
| [IFC](https://github.com/sukjunhwang/IFC)         | 35.2 | 57.2 | 37.5 | -    | -    |          |
| [SeqFormer](https://github.com/wjf5203/SeqFormer) | 40.5 | 62.4 | 43.7 | 36.1 | 48.1 |          |
| Ours                                              | 41.9 | 64.8 | 46.0 | 37.3 | 48.5 |          |


##OVIS

| Model | AP   | AP50 | AP75 | AR1 | AR10 | Pretrain |
|-------|------|------|------|-----|------|----------|
| Ours  | 23.2 | 44.0 | 21.7 |     |      |          |


## Configuration 
We have been inspired by [detectron2](https://github.com/facebookresearch/detectron2) in order to build our configuration system. 
We hope this allows the research community to more easily build upon our method. 
Refer to `src/config.py` to get an overview of all the configuration options available including how the model is built, training and test options.

# Train
We refer to our [docs/TRAIN.md](docs/TRAIN.md) for detailed training instructions.

## Evaluate
To evaluate model's performance, you just need to add the --eval-only argument and set MODEL.WEIGHTS to the checkpoint path via command line.
For example, the following command shows how to validate
```
torchrun --nproc_per_node=1 main.py --config-file configs/devis/devis_R_50.yaml --eval-only DATASETS.VAL_DATASET yt_vis_val_21 MODEL.WEIGHTS /path/to/checkpoint_file
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
