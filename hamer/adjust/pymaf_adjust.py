
from cmath import pi
from math import e
import os
import random
from cv2 import ellipse2Poly
from numpy.core.numeric import base_repr
import torch
import torch.nn as nn
import numpy as np
import json
import joblib

from einops import rearrange
from .body_model import SMPLH
from core.cfgs import cfg
from utils.geometry import   rotation_matrix_to_angle_axis, compute_twist_rotation
from .smpl import SMPL, SMPLX, SMPLX_ALL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, get_model_faces, get_partial_smpl, SMPL_Family
from smplx.lbs import batch_rodrigues
import torch.nn.functional as F
from core import path_config
from os.path import join

import logging
import pdb
logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1


def get_out_smplh(hamer_data=None,hardmo_data=None,smplx=None,body_model=None):
    '''
    Args:
        input_batch: input dictionary, including 
                images: 'img_{part}', for part in body, hand, and face if applicable
                inversed affine transformation for the cropping of hand/face images: '{part}_theta_inv' for part in lhand, rhand, and face if applicable

    Returns:
        out_dict: the list containing the predicted parameters
        vis_feat_list: the list containing features for visualization
    '''
    # 获取 input_batch 的设备
    # device=torch.device('cuda')
    batch_size = 1
    for dic in hamer_data:
        if dic["is_right"] == 0:
            pred_vis_lhand = 1
            pred_orient_lh =dic['mano_global_orient'].reshape(-1,3,3).to('cpu')
            # pred_lhand_rotmat =dic['mano_hand_pose'].reshape(batch_size,-1,3,3).to('cpu')
        if dic["is_right"] == 1:
            pred_vis_rhand = 1
            pred_orient_rh =dic['mano_global_orient'].reshape(-1,3,3).to('cpu')
            # pred_rhand_rotmat =dic['mano_hand_pose'].reshape(batch_size,-1,3,3).to('cpu')

    data2 = hardmo_data
    pred_rotmat_body = batch_rodrigues(torch.from_numpy(hardmo_data['body_pose']).reshape(24,3)).reshape(1,24,3,3).to('cpu')

    pred_shape = torch.from_numpy(data2["betas"].reshape(-1,10)).to('cpu')

    # smpl_global_orient
    # torch.Size([1, 1, 3, 3])
    # smpl_body_pose
    # torch.Size([1, 23, 3, 3]) # batchsize ,24 , 3 ,3
    # body_model = SMPL(
    #                 model_path=SMPL_MODEL_DIR,
    #                 batch_size=64,
    #                 create_transl=False
    #                       )
    
    # if cfg.MODEL.PyMAF.OPT_WRIST:
    #     pred_rotmat_body = rot6d_to_rotmat(pred_pose.reshape(batch_size, -1, 6)) 
   
    pred_gl_body, body_joints = body_model.get_global_rotation(global_orient=pred_rotmat_body[:, 0:1],
                                        body_pose=pred_rotmat_body[:, 1:])
    pred_gl_body = pred_gl_body
    body_joints = body_joints
    
    flip_vector = torch.ones((1, 9), dtype=torch.float32)
    flip_vector[:, [1, 2, 3, 6]] *= -1
    flip_vector = flip_vector.reshape(1, 3, 3)

    #smpl = SMPL_Family(model_type='smpl')

    # smplh =SMPLH(model_path="/data/boning_zhang/PyMAF-X/data/smpl/SMPLH_NEUTRAL.pkl",
    #                     batch_size=batch_size,
    #                     use_pca=False,
    #                     )

                             
  
    pred_gl_lelbow = pred_gl_body[:, 18]
    pred_gl_relbow = pred_gl_body[:, 19]

    # 左手处理
    if 'pred_orient_lh' in locals():  # 检查左手方向是否存在
        target_gl_lwrist = pred_orient_lh
        target_gl_lwrist *= flip_vector.to(target_gl_lwrist.device)
        
        opt_lwrist = torch.bmm(pred_gl_lelbow.transpose(1, 2), target_gl_lwrist)

        # 获取 T-pose 骨骼点
        tpose_joints = smplx.get_tpose(betas=pred_shape.to('cpu'))
        lshoulder_twist_axis = nn.functional.normalize(tpose_joints[:, 18] - tpose_joints[:, 16], dim=1)
        lelbow_twist_axis = nn.functional.normalize(tpose_joints[:, 20] - tpose_joints[:, 18], dim=1)

        lelbow_twist, lelbow_twist_angle = compute_twist_rotation(opt_lwrist, lelbow_twist_axis)

        # 角度限制
        min_angle = -0.4 * float(np.pi)
        max_angle = 0.4 * float(np.pi)

        lelbow_twist_angle[lelbow_twist_angle == torch.clamp(lelbow_twist_angle, min_angle, max_angle)] = 0
        lelbow_twist_angle[lelbow_twist_angle > max_angle] -= max_angle
        lelbow_twist_angle[lelbow_twist_angle < min_angle] -= min_angle

        lelbow_twist = batch_rodrigues(lelbow_twist_axis * lelbow_twist_angle)
        opt_lwrist = torch.bmm(lelbow_twist.transpose(1, 2), opt_lwrist)

        # 过滤结果
        opt_lwrist_filtered = [opt_lwrist[_i] if pred_vis_lhand else pred_rotmat_body[_i, 20] for _i in range(batch_size)]
        opt_lelbow = torch.bmm(pred_rotmat_body[:, 18], lelbow_twist)
        opt_lelbow_filtered = [opt_lelbow[_i] if pred_vis_lhand else pred_rotmat_body[_i, 18] for _i in range(batch_size)]
    else:
        opt_lwrist_filtered = [pred_rotmat_body[_i, 20] for _i in range(batch_size)]
        opt_lelbow_filtered = [pred_rotmat_body[_i, 18] for _i in range(batch_size)]

    # 右手处理
    if 'pred_orient_rh' in locals():  # 检查右手方向是否存在
        target_gl_rwrist = pred_orient_rh
        opt_rwrist = torch.bmm(pred_gl_relbow.transpose(1, 2), target_gl_rwrist)

        # 获取 T-pose 骨骼点
        tpose_joints = smplx.get_tpose(betas=pred_shape.to('cpu'))
        rshoulder_twist_axis = nn.functional.normalize(tpose_joints[:, 19] - tpose_joints[:, 17], dim=1)
        relbow_twist_axis = nn.functional.normalize(tpose_joints[:, 21] - tpose_joints[:, 19], dim=1)

        relbow_twist, relbow_twist_angle = compute_twist_rotation(opt_rwrist, relbow_twist_axis)

        # 角度限制
        min_angle = -0.4 * float(np.pi)
        max_angle = 0.4 * float(np.pi)

        relbow_twist_angle[relbow_twist_angle == torch.clamp(relbow_twist_angle, min_angle, max_angle)] = 0
        relbow_twist_angle[relbow_twist_angle > max_angle] -= max_angle
        relbow_twist_angle[relbow_twist_angle < min_angle] -= min_angle

        relbow_twist = batch_rodrigues(relbow_twist_axis * relbow_twist_angle)
        opt_rwrist = torch.bmm(relbow_twist.transpose(1, 2), opt_rwrist)

        # 过滤结果
        opt_rwrist_filtered = [opt_rwrist[_i] if pred_vis_rhand else pred_rotmat_body[_i, 21] for _i in range(batch_size)]
        opt_relbow = torch.bmm(pred_rotmat_body[:, 19], relbow_twist)
        opt_relbow_filtered = [opt_relbow[_i] if pred_vis_rhand else pred_rotmat_body[_i, 19] for _i in range(batch_size)]
    else:
        opt_rwrist_filtered = [pred_rotmat_body[_i, 21] for _i in range(batch_size)]
        opt_relbow_filtered = [pred_rotmat_body[_i, 19] for _i in range(batch_size)]

    # 最后将所有的结果堆叠到一起
    opt_lwrist = torch.stack(opt_lwrist_filtered)
    opt_rwrist = torch.stack(opt_rwrist_filtered)
    opt_lelbow = torch.stack(opt_lelbow_filtered)
    opt_relbow = torch.stack(opt_relbow_filtered)

    # 更新 pred_rotmat_body
    pred_rotmat_body = torch.cat([pred_rotmat_body[:, :18],
                                   opt_lelbow.unsqueeze(1), opt_relbow.unsqueeze(1),
                                   opt_lwrist.unsqueeze(1), opt_rwrist.unsqueeze(1),
                                   pred_rotmat_body[:, 22:]], 1)

    pred_rotmat = pred_rotmat_body 
    assert pred_rotmat.shape[1] == 24
    smplx_kwargs = {}



    # pred_lhand_rotmat *=  flip_vector.to('cpu').unsqueeze(0)
    # if 'pred_orient_lh' in locals():
    #     smplx_kwargs['left_hand_pose'] = pred_lhand_rotmat.to('cpu')
    # if 'pred_orient_rh' in locals():
    #     smplx_kwargs['right_hand_pose'] = pred_rhand_rotmat.to('cpu')

    # pred_output = smplh(
    #     betas=pred_shape.to('cpu'),
    #     body_pose=pred_rotmat[:, 1:-2].to('cpu'),
    #     global_orient=pred_rotmat[:, 0].unsqueeze(1).to('cpu'),
    #     pose2rot=False,
    #     **smplx_kwargs,
    # )
    

    # pred_joints = pred_output.joints.to('cpu').reshape(-1,3).numpy()

    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1).numpy()

    # print(pose.shape)
    # print(type(pose))
    # print(pose)
    # print(pred_joints.shape)
    # print(type(pred_joints))
    # print(pred_joints)
    data2['body_pose']=pose #(72,)
    # data2['keypoints_3d']=pred_joints #(73,3) 应该为(44,4)

    return data2
# pred_cam, 3
# pred_shape 10
# pose 72
# 'verts'  : pred_vertices, torch.Size([1, 6890, 3])
# 'kp_3d'  : kp_3d,    #torch.Size([1, 49, 3])
# 'rotmat' : pred_rotmat, #torch.Size([1, 24, 3, 3])
# 'pred_shape': pred_shape,#10
# 'smplx_verts': pred_output.smplx_vertices.to(device),  #目前没有 ，返回none
# 'pred_lhand_rotmat': pred_lhand_rotmat,#
# 'pred_rhand_rotmat': pred_rhand_rotmat #torch.Size([15, 3, 3])