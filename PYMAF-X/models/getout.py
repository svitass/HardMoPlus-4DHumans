from cgi import print_arguments
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
from core import constants
from einops import rearrange

from core.cfgs import cfg
from utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d, projection, rotation_matrix_to_angle_axis, rotmat_to_angle, compute_twist_rotation
from .maf_extractor import MAF_Extractor, Mesh_Sampler
from .smpl import SMPL, SMPLX, SMPLX_ALL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, get_model_faces, get_partial_smpl, SMPL_Family
from smplx.lbs import batch_rodrigues
from .hmr import ResNet_Backbone
from .res_module import IUV_predict_layer, Seg_predict_layer, Kps_predict_layer, LimbResLayers
from .hr_module import get_hrnet_encoder
from .pose_resnet import get_resnet_encoder
from utils.imutils import j2d_processing
import torch.nn.functional as F
from utils.keypoints import softmax_integral_tensor
from utils.cam_params import homo_vector
from .attention import get_att_block

from core import path_config
from os.path import join

import logging
import pdb
logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1

def pymaf_net(smpl_mean_params, device=torch.device('cuda'), is_train=True):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        is_train (bool): If True, load ImageNet/COCO pre-trained models
    """
    model = PyMAF(smpl_mean_params, device, is_train)
    return model

def get_out(input_batch={}, J_regressor=None, rw_cam={}):
    '''
    Args:
        input_batch: input dictionary, including 
                images: 'img_{part}', for part in body, hand, and face if applicable
                inversed affine transformation for the cropping of hand/face images: '{part}_theta_inv' for part in lhand, rhand, and face if applicable
        J_regressor: joint regression matrix
        rw_cam: real-world camera information, applied when cfg.MODEL.USE_IWP_CAM is False
    Returns:
        out_dict: the list containing the predicted parameters
        vis_feat_list: the list containing features for visualization
    '''
    # 获取 input_batch 的设备
    device=torch.device('cuda')

    if 'vis_lhand' in input_batch:
        pred_vis_lhand = input_batch['vis_lhand'] > cfg.MODEL.PyMAF.HAND_VIS_TH
    if 'vis_rhand' in input_batch:
        pred_vis_rhand = input_batch['vis_rhand'] > cfg.MODEL.PyMAF.HAND_VIS_TH
    file1="/data/boning_zhang/blend_body_hand_pipline/hamer_out.pkl"
    file2="/data/boning_zhang/blend_body_hand_pipline/4dhuman_out.pkl"
    data1=joblib.load(file1)
    # mano_global_orient
    # torch.Size([2, 1, 3, 3])
    pred_orient_lh =data1['mano_global_orient'][::2].reshape(-1,3,3).to(device)
    pred_orient_rh =data1['mano_global_orient'][1::2].reshape(-1,3,3).to(device)
    data2 = joblib.load(file2)
    pred_rotmat_body =torch.cat((data2["smpl_global_orient"],data2["smpl_body_pose"]),dim=1).to(device)
    batch_size =pred_rotmat_body.shape[0]

    #pred_cam_rh = data1["pred_cam"].to(device)# torch.Size([2, 3])"
    #pred_cam_h = torch.cat([pred_cam_rh[:, 0:1] * 10., pred_cam_rh[:, 1:] / 10.], dim=1).to(device) 
    pred_cam = data2["pred_cam"]
    pred_shape = data2["smpl_betas"].to(device)

    # smpl_global_orient
    # torch.Size([1, 1, 3, 3])
    # smpl_body_pose
    # torch.Size([1, 23, 3, 3]) # batchsize ,24 , 3 ,3
    body_model = SMPL(
                    model_path=SMPL_MODEL_DIR,
                    batch_size=64,
                    create_transl=False
                          )
    
    # if cfg.MODEL.PyMAF.OPT_WRIST:
    #     pred_rotmat_body = rot6d_to_rotmat(pred_pose.reshape(batch_size, -1, 6)) 
   
    pred_gl_body, body_joints = body_model.get_global_rotation(global_orient=pred_rotmat_body[:, 0:1].to('cpu'),
                                        body_pose=pred_rotmat_body[:, 1:].to('cpu'))
    pred_gl_body = pred_gl_body.to(device)
    body_joints = body_joints.to(device)
    
    flip_vector = torch.ones((1, 9), dtype=torch.float32)
    flip_vector[:, [1, 2, 3, 6]] *= -1
    flip_vector = flip_vector.reshape(1, 3, 3)


    smpl = SMPL_Family(model_type='smplx', 
                                model_path=join(path_config.SMPL_MODEL_DIR, 'SMPLX_NEUTRAL_2020.npz'),
                                num_expression_coeffs=cfg.MODEL.N_EXP)
    if cfg.MODEL.PyMAF.OPT_WRIST:
        pred_gl_lelbow = pred_gl_body[:, 18]
        pred_gl_relbow = pred_gl_body[:, 19]

        target_gl_lwrist = pred_orient_lh
        target_gl_lwrist *= flip_vector.to(target_gl_lwrist.device)
        target_gl_rwrist = pred_orient_rh
        #pdb.set_trace()
        opt_lwrist = torch.bmm(pred_gl_lelbow.transpose(1, 2), target_gl_lwrist)
        opt_rwrist = torch.bmm(pred_gl_relbow.transpose(1, 2), target_gl_rwrist)

        if cfg.MODEL.PyMAF.ADAPT_INTEGR:
           
            tpose_joints = smpl.get_tpose(betas=pred_shape.to('cpu'))
            tpose_joints = tpose_joints.to(device)
            lshoulder_twist_axis = nn.functional.normalize(tpose_joints[:, 18] - tpose_joints[:, 16], dim=1)
            rshoulder_twist_axis = nn.functional.normalize(tpose_joints[:, 19] - tpose_joints[:, 17], dim=1)
            lelbow_twist_axis = nn.functional.normalize(tpose_joints[:, 20] - tpose_joints[:, 18], dim=1)
            relbow_twist_axis = nn.functional.normalize(tpose_joints[:, 21] - tpose_joints[:, 19], dim=1)

            lelbow_twist, lelbow_twist_angle = compute_twist_rotation(opt_lwrist, lelbow_twist_axis)
            relbow_twist, relbow_twist_angle = compute_twist_rotation(opt_rwrist, relbow_twist_axis)

            min_angle = -0.4 * float(np.pi)
            max_angle = 0.4 * float(np.pi)

            lelbow_twist_angle[lelbow_twist_angle==torch.clamp(lelbow_twist_angle, min_angle, max_angle)]=0
            relbow_twist_angle[relbow_twist_angle==torch.clamp(relbow_twist_angle, min_angle, max_angle)]=0
            lelbow_twist_angle[lelbow_twist_angle > max_angle] -= max_angle
            lelbow_twist_angle[lelbow_twist_angle < min_angle] -= min_angle
            relbow_twist_angle[relbow_twist_angle > max_angle] -= max_angle
            relbow_twist_angle[relbow_twist_angle < min_angle] -= min_angle

            lelbow_twist = batch_rodrigues(lelbow_twist_axis * lelbow_twist_angle)
            relbow_twist = batch_rodrigues(relbow_twist_axis * relbow_twist_angle)

            opt_lwrist = torch.bmm(lelbow_twist.transpose(1, 2), opt_lwrist)
            opt_rwrist = torch.bmm(relbow_twist.transpose(1, 2), opt_rwrist)

            # left elbow: 18
            opt_lelbow = torch.bmm(pred_rotmat_body[:, 18], lelbow_twist)
            # right elbow: 19
            opt_relbow = torch.bmm(pred_rotmat_body[:, 19], relbow_twist)

            if cfg.MODEL.PyMAF.PRED_VIS_H :
                opt_lwrist_filtered = [opt_lwrist[_i] if pred_vis_lhand[_i] else pred_rotmat_body[_i, 20] for _i in range(batch_size)]
                opt_rwrist_filtered = [opt_rwrist[_i] if pred_vis_rhand[_i] else pred_rotmat_body[_i, 21] for _i in range(batch_size)]
                opt_lelbow_filtered = [opt_lelbow[_i] if pred_vis_lhand[_i] else pred_rotmat_body[_i, 18] for _i in range(batch_size)]
                opt_relbow_filtered = [opt_relbow[_i] if pred_vis_rhand[_i] else pred_rotmat_body[_i, 19] for _i in range(batch_size)]

                opt_lwrist = torch.stack(opt_lwrist_filtered)
                opt_rwrist = torch.stack(opt_rwrist_filtered)
                opt_lelbow = torch.stack(opt_lelbow_filtered)
                opt_relbow = torch.stack(opt_relbow_filtered)

                pred_rotmat_body = torch.cat([pred_rotmat_body[:, :18],
                                                    opt_lelbow.unsqueeze(1), opt_relbow.unsqueeze(1), 
                                                    opt_lwrist.unsqueeze(1), opt_rwrist.unsqueeze(1), 
                                                    pred_rotmat_body[:, 22:]], 1)
    pred_rotmat = pred_rotmat_body 
    assert pred_rotmat.shape[1] == 24
    smplx_kwargs = {}

    


        # if self.full_body_mode:
    pred_lhand_rotmat =data1['mano_hand_pose'][0::2].reshape(batch_size,-1,3,3)
    pred_rhand_rotmat =data1['mano_hand_pose'][1::2].reshape(batch_size,-1,3,3)#这样不失去维度
    pred_lhand_rotmat *=  flip_vector.to(pred_lhand_rotmat.device).unsqueeze(0)
    # if cfg.MODEL.HAND_PCA_ON :
    #     pred_lhand_rotmat, pred_rhand_rotmat = hand_pca_filter(pred_lhand_rotmat, pred_rhand_rotmat)
    smplx_kwargs['left_hand_pose'] = pred_lhand_rotmat.to('cpu')
    smplx_kwargs['right_hand_pose'] = pred_rhand_rotmat.to('cpu')
    # smplx_kwargs['jaw_pose'] = pred_face_rotmat[:, 0:1]
    # smplx_kwargs['leye_pose'] = pred_face_rotmat[:, 1:2]
    # smplx_kwargs['reye_pose'] = pred_face_rotmat[:, 2:3]
    # smplx_kwargs['expression'] = pred_exp   
    #pdb.set_trace()
    pred_output = smpl(
        betas=pred_shape.to('cpu'),
        body_pose=pred_rotmat[:, 1:].to('cpu'),
        global_orient=pred_rotmat[:, 0].unsqueeze(1).to('cpu'),
        pose2rot=False,
        **smplx_kwargs,
    )
    #pdb.set_trace()
    pred_vertices = pred_output.vertices.to(device)
    pred_joints = pred_output.joints.to(device)

    if J_regressor is not None:
        kp_3d = torch.matmul(J_regressor, pred_vertices)
        pred_pelvis = kp_3d[:, [0], :].clone()
        kp_3d = kp_3d[:, constants.H36M_TO_J14, :]
        kp_3d = kp_3d - pred_pelvis
    else:
        kp_3d = pred_joints
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
    out_put={}
    #pdb.set_trace()
    out_put.update({
                    'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
                    'verts'  : pred_vertices,
                    'kp_3d'  : kp_3d,
                    'rotmat' : pred_rotmat,
                    'pred_shape': pred_shape,
                    'smplx_verts': pred_output.smplx_vertices.to(device),
                    'pred_lhand_rotmat': pred_lhand_rotmat,
                    'pred_rhand_rotmat': pred_rhand_rotmat,})
    return out_put
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