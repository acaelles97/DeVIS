## Visdom
Monitoring of the training/evaluation progress is possible via command line as well as [Visdom](https://github.com/fossasia/visdom). 
For the latter, a Visdom server must be running at `VISDOM_PORT=8090` and `VISDOM_SERVER=http://localhost`. 
To deactivate Visdom logging set `VISDOM_ON=False`.

## Train
We provide configurations files under `configs/deformable_mask_head/` and `configs/devis/` to train Deformable Mask-Head  and DeVIS respectively.
In order to launch a training you just need to simply specify the number of GPUS using `--nproc_per_node` and the corresponding config file after `--config-file`. For instance, the command for training YT-VIS 2019 model with 4GPUs is as following:

```
torchrun --nproc_per_node=4 main.py --config-file configs/devis/devis_R_50_YT-19.yaml
```
User can also override config file parameters by passing the new KEY VALUE pair. 
For instance, to double the default lr:

```
torchrun --nproc_per_node=4 main.py --config-file configs/devis/devis_R_50_YT-19.yaml SOLVER.BASE_LR 0.0002
```

## Model zoo

| Dataset        | AP     | AP50   | AP75   | AR1   | AR10   | Training<br/> GPU hours \* | Max GPU <br/>memory    | URL                                                                                                                                               |
|----------------|--------|--------|--------|-------|--------|----------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| COCO           | ------ | ------ | ------ | ----- | ------ | ------------------------   | ---------------------- | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/deformable_mask_head/deformable_mask_head_R_50.yaml) <br/>[log]() <br/>[model]() |
| YouTube-VIS 19 | ------ | ------ | ------ | ----- | ------ | ------------------------   | ---------------------- | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_R_50_YT-19.yaml) <br/>[log]() <br/>[model]()                         |
| YouTube-VIS 21 | ------ | ------ | ------ | ----- | ------ | ------------------------   | ---------------------- | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_R_50_YT-21.yaml) <br/>[log]() <br/>[model]()                         |
| OVIS           | 23.2   | 45.2   | 22.5   | 11.8  | 27.9   | 116                        | 24GB                   | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_R_50_OVIS.yaml) <br/>[log](https://vision.in.tum.de/webshare/u/cad/model_zoo/ovis/log.out) <br/>[model](https://vision.in.tum.de/webshare/u/cad/model_zoo/ovis/r50_devis_ovis.zip)                          |

## Ablations
We also provide configuration file to run all the ablation studies presented on Table 1:

| Method                        | Clip size | K_temp | Features scales | AP   | Training<br/> GPU hours\* | Max GPU <br/>memory | URL                                                                                                                                 |
|-------------------------------|-----------|--------|-----------------|------|--------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Deformable VisTR              | 36        | 4      | 1               | 33.0 | 400                      | 300                 | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation0_deformable_vistr.yaml) <br/>log              |
| Deformable VisTR              | 36        | 0      | 1               | 33.0 | 400                      | 300                 | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation1_deformable_vistr_wo_temp_conn.yaml) <br/>log |
| DeVIS                         | 6         | 4      | 4               |      |                          |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation2_single-scale.yaml) <br/>log                  |
| +increase spatial inputs      | 6         | 4      | 4               |      |                          |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation3_increased-spatial-inputs.yaml) <br/>log      |
| +instance aware obj. queries  | 6         | 4      | 4               |      |                          |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation4_instance-aware.yaml) <br/>log                |
| +multi-scale mask head        | 6         | 4      | 4               |      |                          |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation5_multi-scale_mask-head.yaml) <br/>log         |
| +multi-cue clip tracking      | 6         | 4      | 4               |      | --                       | --                  | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_ablation6_TEST_multi-cue_tracking.yaml)                |
| +aux. loss weighting          | 6         | 4      | 4               |      |                          |                     | [config](https://github.com/acaelles97/DeVIS/blob/master/configs/devis/devis_R_50_YT-19.yaml)  <br/>log                             |


*Training GPU hours measured on a RTX A6000 GPU, includes validation

## Validation during training
We support evaluation during training for VIS datasets despite GT annotations not available.
Results will be saved into `TEST.SAVE_PATH` folder, created inside `OUTPUT_DIR`.
Users can set `EVAL_PERIOD` to select the interval between validations (0 to disable it) 
Additionally, `START_EVAL_EPOCH` allows selecting at which epoch start considering `EVAL_PERIOD` in order to omit first epochs.