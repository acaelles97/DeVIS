## Visdom
Monitoring of the training/evaluation progress is possible via command line as well as [Visdom](https://github.com/fossasia/visdom). 
For the latter, a Visdom server must be running at `VISDOM_PORT=8090` and `VISDOM_SERVER=http://localhost`. 
To deactivate Visdom logging set `VISDOM_ON=False`.

TODO: SPECIFY REQUIREMENTS FOR EACH TRAINING

## Train
We provide configurations files to train Deformable DeTR `configs/deformable_detr/deformable_detr_R_50.yaml`, Deformable Mask-Head `configs/deformable_mask_head/deformable_mask_head_R_50.yaml` and DeVIS `configs/devis/devis_R_50.yaml`.

For instance, the command to train DeVIS on YT-21 with 4GPUs is as following:

```
torchrun --nproc_per_node=4 main.py --config-file configs/devis/devis_R_50.yaml DATASETS.TRAIN_DATASET yt_vis_train_21 DATASETS.VAL_DATASET yt_vis_val_21
```
As shown above, user can override config-file parameters by passing the new KEY VALUE pair.    


| Method                        | Clip size | K_temp | Features scales | AP   | Training<br/> GPU hours | Max GPU <br/>memory | URL                                                                                                                                 |
|-------------------------------|-----------|--------|-----------------|------|-------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Deformable VisTR              | 36        | 4      | 1               | 33.0 | 400                     | 300                 | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation0_deformable_vistr.yaml) <br/>log              |
| Deformable VisTR              | 36        | 0      | 1               | 33.0 | 400                     | 300                 | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation1_deformable_vistr_wo_temp_conn.yaml) <br/>log |
| DeVIS                         | 6         | 4      | 4               |      |                         |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation2_single-scale.yaml) <br/>log                  |
| +increase spatial inputs      | 6         | 4      | 4               |      |                         |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation3_increased-spatial-inputs.yaml) <br/>log      |
| +instance aware obj. queries  | 6         | 4      | 4               |      |                         |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation4_instance-aware.yaml) <br/>log                |
| +multi-scale mask head        | 6         | 4      | 4               |      |                         |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation5_multi-scale_mask-head.yaml) <br/>log         |
| +multi-cue clip tracking      | 6         | 4      | 4               |      |                         |                     | [config]()                                                                                                                          |
| +aux. loss weighting          | 6         | 4      | 4               |      |                         |                     | [config]()                                                                                                                          |

# Validation during training
