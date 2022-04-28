# Installation

1. Clone and enter this repository:
    ```
    git clone git@github.com:acaelles97/DeVIS.git
    cd DeVIS
    ```
2. Install packages for Python 3.8:
   1. Install PyTorch 1.11.0 and torchvision 0.12.0 from [here](https://pytorch.org/get-started/locally/). The tested CUDA version is 11.3.0 
   2. `pip3 install -r requirements.txt`
   3. Install [youtube-vis](https://github.com/youtubevos/cocoapi) api
   4. Install MultiScaleDeformableAttention package: `python src/models/ops/setup.py build_ext install`

If you experience problems installing youtube-vis api, we recommend cloning the repo and adding its path to the import part as follows:


## Dataset preparation
First step is to download and extract each dataset: [COCO](https://cocodataset.org/#home), [YT-19](https://youtube-vos.org/dataset/vis/), [YT-21](https://youtube-vos.org/dataset/vis/) & [OVIS](http://songbai.site/ovis/)
We expect the following organization for  training. 
User must set `DATASETS.DATA_PATH` to the root data path. 
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

## Download weights