import os
from pathlib import Path
cur_project_path = Path(__file__).resolve().parent.parent.parent.parent
print("cur_project_path:", cur_project_path)
import sys
sys.path.append(cur_project_path)
import pickle
import numpy as np
import glob
import torch
from hmr2.utils.geometry import aa_to_rotmat, perspective_projection
import yaml
from hmr2.models.smpl_wrapper import SMPL
"""
{
    "imgname": 图片名list[imgname1, imgname2, ..., imgnamen], 
    "center": (nframes, 1, 2), 
    "scale": (nframes, 1, 2), 
    "body_pose" : (nframes, 72),
    "has_body_pose": (nframes,),
    "betas": (nframes, 10), 
    "has_betas": (nframes, ), 
    "body_keypoints_2d": (nframes, 25, 3),
    "extra_keypoints_2d": (nframes, 19, 3)
    "body_keypoints_3d": (nframes, 25, 3),
    "extra_keypoints_3d": (nframes, 19, 3)
}
"""
eval_data_root = "/data/boning_zhang/Freeman/freeman_sideview"
eval_jpg_list = glob.glob(eval_data_root + "/*.jpg")
npz_path = "/data/boning_zhang/Freeman/freeman_sideview.npz"
npz_data_dict = {}
imgname_list = []
center_list = []
scale_list = []
bodypose_list = []
has_bodypose_list = []
betas_list = []
has_betas_list = []
kps2d_list = []
extra_kps2d_list = []
kps3d_list = []
extra_kps3d_list = []
smpl_cfg = {}
smpl_cfg['model_path'] = 'data/smpl'
smpl_cfg['gender'] = 'neutral'
smpl_cfg['num_body_joints'] = 23
smpl_cfg['joint_regressor_extra'] = 'data/SMPL_to_J19.pkl'
smpl_cfg['mean_params'] = 'data/smpl_mean_params.npz'
smpl = SMPL(**smpl_cfg)
for jpg_path in eval_jpg_list:
    imgname = os.path.basename(jpg_path)
    pyd_path = eval_data_root + "/" + imgname.replace(".jpg", ".data.pyd")
    with open(pyd_path, "rb") as file:
        data = pickle.load(file)[0]
        center = data["center"]
        scale = data["scale"]
        body_pose = data["body_pose"]
        has_body_pose = data["has_body_pose"]
        betas = data["betas"]
        has_betas = data["has_betas"]
        body_keypoints_2d = data["keypoints_2d"][:25, :]
        extra_keypoints_2d = data["keypoints_2d"][25:, :]
        pred_smpl_params = {}
        pred_smpl_params["body_pose"] = aa_to_rotmat(torch.from_numpy(body_pose[3:].reshape(-1, 3))).view(-1, 3,
                                                                                                          3).unsqueeze(
            0)
        pred_smpl_params["global_orient"] = aa_to_rotmat(torch.from_numpy(body_pose[:3].reshape(-1, 3))).view(-1, 3,
                                                                                                              3).unsqueeze(
            0)
        pred_smpl_params["betas"] = torch.from_numpy(betas.reshape(-1, 10))
        smpl_output = smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = torch.cat(
            (smpl_output.joints.squeeze(0), torch.ones(smpl_output.joints.squeeze(0).shape[0], 1)),
            dim=-1).cpu().numpy()

        body_keypoints_3d = pred_keypoints_3d[:25, :]
        extra_keypoints_3d = pred_keypoints_3d[25:, :]
    imgname_list.append(imgname)
    center_list.append(center)
    scale_list.append(scale)
    bodypose_list.append(body_pose)
    has_bodypose_list.append(has_body_pose)
    betas_list.append(betas)
    has_betas_list.append(has_betas)
    kps2d_list.append(body_keypoints_2d)
    extra_kps2d_list.append(extra_keypoints_2d)
    kps3d_list.append(body_keypoints_3d)
    extra_kps3d_list.append(extra_keypoints_3d)

npz_data_dict["imgname"] = imgname_list
npz_data_dict["center"] = np.array(center_list)[:, np.newaxis, :]   # (nframes, 1, 2)
npz_data_dict["scale"] = np.array(scale_list)[:, np.newaxis, :]  # (nframes, 1, 2)
npz_data_dict["body_pose"] = np.array(bodypose_list)  # (nframes, 72)
npz_data_dict["has_body_pose"] = np.array(has_bodypose_list)  # (nframes,)
npz_data_dict["betas"] = np.array(betas_list)  # (nframes, 10)
npz_data_dict["has_betas"] = np.array(has_betas_list)  # (nframes, ),
npz_data_dict["body_keypoints_2d"] = np.array(kps2d_list)
npz_data_dict["extra_keypoints_2d"] = np.array(extra_kps2d_list)
npz_data_dict["body_keypoints_3d"] = np.array(kps3d_list)
npz_data_dict["extra_keypoints_3d"] = np.array(extra_kps3d_list)
# # 保存成npz文件
np.savez(npz_path, **npz_data_dict)
print('Finished!')
