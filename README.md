# HardMo++: A Large-Scale Hardcase Dataset for Motion Capture

## Get Start

This code was tested on `NVIDIA V100` and requires:

* conda3 or miniconda3
* python 3.10+

Create a conda virtual environment and activate it.
```shell
conda env create -f environment.yml
conda activate human4d
```

## Quick Start
1. Get Training Dataset 
Download training dataset from [BaiduYun Cloud](xxx) and put them into ./hmr2_training_data/.
2. Get Evaluation Dataset
Download evaluation dataset from [BaiduYun Cloud](xxx) and put them into ./hmr2_evaluation_data/.
3. Get Pre-trained Models
pre-trained models from [Google Cloud](https://drive.google.com/drive/folders/1o3h0DHEz5gKG-9cTdl3lUEwjwW51Ay81?usp=sharing) and put them into ./data/.
4. The expected file structure is as follows:
```text
HardMoPlus-4DHumans
├── data
│   ├── SMPL_to_J19.pkl
│   ├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
│   ├── smpl_mean_params.npz
│   └── smpl
│       ├── SMPL_FEMALE.pkl
│       ├── SMPL_MALE.pkl
│       └── SMPL_NEUTRAL.pkl
│   └── baseline
│       ├── hmr2_hardmoplus
│       ├── hmr2a
│       ├── hmr2b
│       ├── hmr_hardmo
│       ├── pro_hmr
│       └── hmr2_hardmo
│           ├── dataset_config.yaml
│           ├── model_config.yaml
│           └── checkpoints
│               └── epoch=25-step=85000.ckpt
├── hmr2_training_data
│   ├── amass_poses_hist100_SMPL+H_G.npy
│   ├── cmu_mocap.npz
│   ├── hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
│   ├── pose_hrnet_w48_256x192.pth
│   ├── vitpose_backbone.pth
│   └── dataset_tars
│       ├── freeman-train
│       ├── freeman-val
│       └── hardmo-train
├── hmr2_evaluation_data
│   ├── freeman_eval
│   │   ├── 20221018_b2bead8274_subj38_001851_c05.jpg
│   │   └── 20221018_b2bead8274_subj38_001851_c06.data.pyd
│   ├── freeman_eval.npz
│   ├── freeman_sideview
│   └── freeman_sideview.npz
├── scripts
└── train_freeman.py
```

##  Train and Evaluation
### 1. Train
```shell
bash scripts/train.sh
```

### 2. Demo
```shell
bash scripts/demo.sh
```
### 3. Evaluation on freeman benchmark
```shell
bash scripts/eval_freeman.sh
```

### 4. Evaluation on sideview benchmark
```shell
bash scripts/eval_freeman_sideview.sh
```

##  Dataset preprocess
### 1. Optimize motion from hardmo subset with 2d keypoints guidance
```shell
python data_processes/hardmo_motion_optimize.py 
```
### 2. Optimize motion from freeman subset with 2d keypoints guidance and multiview guidance
```shell
python data_processes/freeman_motion_optimize.py
```
### 3. Merge the hand and body parts using an elbow rotation compensation strategy
First, you need to configure the environments separately following the instructions for [HaMer](https://github.com/geopavlakos/hamer) and [PyMAF-X](https://github.com/HongwenZhang/PyMAF-X), naming the environments 'hamer' and 'pymaf-x' respectively. Then, run the following code.
```shell
cd HandBody-fusion
bash blend.sh
```

## License
The dataset is under CC BY-SA 4.0 license.


## Acknowledgments

We want to thank the following contributors that our code is based on:

[4D-Humans](https://github.com/shubham-goel/4D-Humans), [PRO-HMR](https://github.com/nkolot/ProHMR).

## Citing
If you find this code useful for your research, please consider citing the following paper:
