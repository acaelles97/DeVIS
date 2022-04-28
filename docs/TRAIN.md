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


| Method                        | Clip size | Features scales | AP  | Training<br/> GPU hours | Max GPU <br/>memory   | URL                                                                   |
|-------------------------------|-----------|-----------------|-----|-------------------------|-----------------------|-----------------------------------------------------------------------|
| Deformable VisTR              | 36        | 1               | 1   | 33.0                    | 300                   | [config](configs/devis/devis_ablation0_deformable_vistr.yaml)<br/>log |
| DeVIS                         | 6         | 1               |     |                         |                       | config<br/>log                                                        |
| +increase spatial inputs      | 6         | 4               |     |                         |                       | config<br/>log                                                        |
| +instance aware obj. queries  | 6         | 4               |     |                         |                       | config<br/>log                                                        |
| +multi-scale mask head        | 6         | 4               |     |                         |                       | config<br/>log                                                        |
| +multi-cue clip tracking      | 6         | 4               |     |                         |                       | config<br/>log                                                        |
| +aux. loss weighting          | 6         | 4               |     |                         |                       | config<br/>log                                                        |

# Validation during training
