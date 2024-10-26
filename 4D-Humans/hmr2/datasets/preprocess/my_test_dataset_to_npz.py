from mmpose.apis import MMPoseInferencer
import argparse
import os
from tqdm import tqdm
import numpy as np
import pickle
from multiprocessing import Process
import multiprocessing as mp
import torch
import sys
sys.path.append("/data/jiaqi_liao/4D-Humans")
from ultralytics import YOLO
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.utils.geometry import aa_to_rotmat, perspective_projection
from hmr2.utils.rotation_convert import matrix_to_axis_angle
import cv2
from hmr2.models.losses import Keypoint2DLoss
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hmr2.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
from hmr2.models.smpl_wrapper import SMPL
import re
from PIL import Image
from hmr2.models.smpl_wrapper import SMPL
from hmr2.datasets.smplh_prob_filter import poses_check_probable, load_amass_hist_smooth
import yaml

# 打开并读取配置文件

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path
def work(pyd_dataset_path, out_path, smpl):
    # scale factor
    scaleFactor = 1.2
    imgnames_, scales_, centers_, extra_keypoints_2d_, extra_keypoints_3d_, body_keypoints_2d_, body_keypoints_3d_, hand_keypoints_2d_= [], [], [], [], [], [], [], []
    poses_, betas_= [], []

    for pyd_path in tqdm(pyd_dataset_path):
        base_image_name = '/'.join(pyd_path.split('/')[-2:]).replace('.pyd', '.jpeg').replace('_10000000', '')
        print(base_image_name)

        if not os.path.exists(pyd_path):
            continue

        with open(pyd_path, "rb") as f:
            data = pickle.load(f)
        imagenames = base_image_name
        center = data['center']
        scale = data['scale']
        body_keypoints_2d = data["keypoints_2d"][:25,:]
        extra_keypoints_2d = data["keypoints_2d"][25:,:]
        hand_keypoints_2d = data["hand_keypoints_2d"]

        body_pose = data['body_pose']
        has_body_pose = data['has_body_pose']

        betas = data['betas']
        has_betas = data['has_betas']
        
        pred_smpl_params = {}
        pred_smpl_params["body_pose"] = aa_to_rotmat(torch.from_numpy(body_pose[3:].reshape(-1, 3))).view(-1, 3, 3).unsqueeze(0)
        pred_smpl_params["global_orient"] =  aa_to_rotmat(torch.from_numpy(body_pose[:3].reshape(-1, 3))).view(-1, 3, 3).unsqueeze(0)
        pred_smpl_params["betas"] =  torch.from_numpy(betas.reshape(-1, 10))
        smpl_output= smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d =  torch.cat((smpl_output.joints.squeeze(0), torch.ones(smpl_output.joints.squeeze(0).shape[0], 1)), dim=-1).cpu().numpy()
        

        body_keypoints_3d = pred_keypoints_3d[:25,:]
        extra_keypoints_3d = pred_keypoints_3d[25:,:]

        imgnames_.append(imagenames)
        centers_.append(center)
        scales_.append(scale)
        poses_.append(body_pose)
        betas_.append(betas)
        body_keypoints_2d_.append(body_keypoints_2d)
        extra_keypoints_2d_.append(extra_keypoints_2d)
        body_keypoints_3d_.append(body_keypoints_3d)
        extra_keypoints_3d_.append(extra_keypoints_3d)
        hand_keypoints_2d_.append(hand_keypoints_2d)

    np.savez(out_path,
        imgname=imgnames_,
        center=centers_,
        scale=scales_,
        body_pose=poses_,
        has_body_pose=np.ones(len(poses_)),
        betas=betas_,
        has_betas=np.ones(len(betas_)),
        body_keypoints_2d=body_keypoints_2d_,
        extra_keypoints_3d=extra_keypoints_3d_,
        body_keypoints_3d=body_keypoints_3d_,
        extra_keypoints_2d=extra_keypoints_2d_,
        hand_keypoints_2d=hand_keypoints_2d_
        )

    #

        

def main(image_folder:str, out_path:str):
    ###这个是image的文件夹，然后我们直接得出pyd_path
    image_paths = findAllFile(image_folder)
    with open('/data/jiaqi_liao/4D-Humans/test.yaml', 'r') as file:
        config = yaml.safe_load(file)
    smpl_cfg = {k.lower(): v for k,v in dict(config["SMPL"]).items()}
    smpl = SMPL(**smpl_cfg)
    work(image_paths, out_path, smpl)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='/data/jiaqi_liao/dance_pyd_source/ballet/5')
    parser.add_argument('--out_path', type=str, default='/data/jiaqi_liao/4D-Humans/hard_case_pipeline/test.npz')
    args = parser.parse_args()
    main(args.image_folder, args.out_path)
