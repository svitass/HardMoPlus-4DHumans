import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))

import numpy as np
import torch
torch.manual_seed(0)

from score_hmr.utils.motion_retarget import target_idx
from score_hmr.utils import *
from score_hmr.datasets.utils import get_example
from score_hmr.configs import dataset_config, model_config
from score_hmr.models.model_utils import load_diffusion_model, load_pare
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from score_hmr.datasets.videt_dataset import ViTDetDataset
import os
import cv2
import pickle
from score_hmr.utils.constants import DEFAULT_IMG_SIZE, DEFAULT_MEAN, DEFAULT_STD, FLIP_KEYPOINT_PERMUTATION
import copy
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hmr2
from pathlib import Path
from tqdm import tqdm
import glob
import multiprocessing as mp
import argparse

def convert_targetbody_to_openpose25(targetbody_keypoints):
    targetbody_to_openpose_map = [15, 7, 10, 12, 14, 9, 11, 13, 0, 2, 4, 6, 1, 3, 5, 17, 16, 19, 18, 20, 21, 22, 23, 24, 25]
    openpose_keypoints = targetbody_keypoints[targetbody_to_openpose_map, :]
    return openpose_keypoints

class BatchedImageDataset:
    def __init__(self, kps2d_folder, img_folder, subset_mname, detector, model_cfg_hmr2, hmr2_model, device):
        super(BatchedImageDataset, self).__init__()
        self.keypoints2d_dir = kps2d_folder
        self.image_dir = img_folder
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD
        self.img_size = DEFAULT_IMG_SIZE
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)
        self.filelist = sorted(subset_mname)
        self.model_cfg_hmr2 = model_cfg_hmr2
        self.hmr2_model = hmr2_model
        self.device = device
        self.detector = detector

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx:int):
        filename = self.filelist[idx]
        views = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08']
        # variable in 2d keypoints scorehmr
        pred_cam_t_view = []
        center_view = []
        img_size_view = []
        scorehmr_kps2d_img_view = []  # (1, 3, 244, 244)
        scorehmr_kps2d_personid_view = []  # (1)
        scorehmr_kps2d_boxcenter_view = []  # (1, 2)
        scorehmr_kps2d_imgsize_view = []  # (1, 2)
        scorehmr_kps2d_view = []  # (1, 63, 3)
        # variable in multiview scorehmr
        keypoints2d_view = []
        img_patch = []
        pred_pose_view = []
        scorehmr_pred_betas_view = []
        # variable in render mesh
        all_verts_view = []
        all_cam_t_view = []
        scaled_focal_length_view = []
        hmr2_img_size_view = []
        img_cv2_view = []
        imgname_view = []
        pred_betas_view = []
        global_orient_view = []
        body_pose_view = []
        while len(scorehmr_kps2d_img_view) != 8:
            for cid, view in enumerate(views):
                imgname = filename + f"_{view}"
                imgname_view.append(imgname)
                img_path = self.image_dir + "/" + filename + f"_{view}.jpg"
                kps2d_name = "_".join(filename.split("_")[0:3])  # 20220926_f469e7df06_subj11
                frame_id = int(filename.split("_")[3])
                keypoints2d_path = self.keypoints2d_dir + "/" + kps2d_name + ".npy"
                keypoints2d_data = np.load(keypoints2d_path, allow_pickle=True)
                kps2d = keypoints2d_data[0]['keypoints2d'][cid, frame_id, target_idx]
                kps2d = convert_targetbody_to_openpose25(kps2d)
                kps2d_extra = np.zeros((19, 3))
                keypoints_2d = np.concatenate((kps2d, kps2d_extra), axis=0).astype(np.float32)  # (44, 3)
                center = keypoints2d_data[0]['center'][cid, frame_id].astype(np.float32)
                scale = keypoints2d_data[0]['scale'][cid, frame_id].astype(np.float32)
                keypoints2d_view.append(keypoints_2d)
                x_min = np.min(keypoints_2d[:, 0])
                y_min = np.min(keypoints_2d[:, 1])
                x_max = np.max(keypoints_2d[:, 0])
                y_max = np.max(keypoints_2d[:, 1])
                # use hmr2 to predict a initialize mesh
                img_cv2 = cv2.imread(img_path)
                det_out = self.detector(img_cv2)  # Detect humans in image
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                hmr2_dataset = ViTDetDataset(self.model_cfg_hmr2, img_cv2, boxes)
                hmr2_dataloader = torch.utils.data.DataLoader(hmr2_dataset, batch_size=8, shuffle=False, num_workers=0)
                all_verts = []
                all_cam_t = []
                hmr2_out = None
                for hmr2_batch in hmr2_dataloader:
                    hmr2_batch = recursive_to(hmr2_batch, self.device)
                    with torch.no_grad():
                        hmr2_out = self.hmr2_model(hmr2_batch)
                        hmr2_pred_cam = hmr2_out['pred_cam']
                        hmr2_box_center = hmr2_batch['box_center'].float()
                        hmr2_box_size = hmr2_batch['box_size'].float()
                        hmr2_img_size = hmr2_batch['img_size'].float()
                        scaled_focal_length = self.model_cfg_hmr2.EXTRA.FOCAL_LENGTH / self.model_cfg_hmr2.MODEL.IMAGE_SIZE * hmr2_img_size.max()
                        pred_cam_t_full = cam_crop_to_full(hmr2_pred_cam, hmr2_box_center, hmr2_box_size, hmr2_img_size,
                                                           scaled_focal_length).detach().cpu().numpy()
                        # find the target person id by calculating IOU
                        idx = 0
                        max_iou = -float('inf')
                        for i, (hmr2_x_min, hmr2_y_min, hmr2_x_max, hmr2_y_max) in enumerate(boxes):
                            x_min_overlap = max(x_min, hmr2_x_min)
                            y_min_overlap = max(y_min, hmr2_y_min)
                            x_max_overlap = min(x_max, hmr2_x_max)
                            y_max_overlap = min(y_max, hmr2_y_max)
                            overlap_width = max(0, x_max_overlap - x_min_overlap)
                            overlap_height = max(0, y_max_overlap - y_min_overlap)
                            area_overlap = overlap_width * overlap_height
                            area_box1 = (x_max - x_min) * (y_max - y_min)
                            area_box2 = (hmr2_x_max - hmr2_x_min) * (hmr2_y_max - hmr2_y_min)
                            area_union = area_box1 + area_box2 - area_overlap
                            iou = area_overlap / area_union if area_union > 0 else 0  # IoU area
                            if iou > max_iou:
                                max_iou = iou
                                idx = i
                        verts = hmr2_out['pred_vertices'][idx].detach().cpu().numpy()
                        all_verts.append(verts)
                        cam_t = pred_cam_t_full[idx]
                        all_cam_t.append(cam_t)
                        scaled_focal_length_view.append(scaled_focal_length)
                        hmr2_img_size_view.append(hmr2_img_size[idx])
                        img_cv2_view.append(img_cv2)
                        pred_betas = hmr2_out['pred_smpl_params']['betas'][idx]  # for render
                        pred_betas_view.append(pred_betas)
                        global_orient = hmr2_out['pred_smpl_params']['global_orient'][idx]
                        global_orient_view.append(global_orient)
                        body_pose = hmr2_out['pred_smpl_params']['body_pose'][idx]
                        body_pose_view.append(body_pose)
                        img_size_view.append(hmr2_img_size[idx].detach().cpu().numpy())
                        # for keypoints2d render
                        kps2d_pred_cam_t_full = cam_crop_to_full(hmr2_pred_cam, hmr2_box_center, hmr2_box_size,
                                                                 hmr2_img_size)
                        pred_cam_t_view.append(kps2d_pred_cam_t_full[idx].detach().cpu().numpy())
                    # for 2d keypoints scorehmr
                    scorehmr_keypoints_2d = np.array([keypoints_2d[:25, ...]])
                    boxes = np.array([boxes[idx]])
                    scorehmr_kps2d_dataset = ViTDetDataset(self.model_cfg_hmr2, img_cv2, boxes,
                                                           body_keypoints=scorehmr_keypoints_2d, is_hmr2=False)
                    scorehmr_kps2d_dataloader = torch.utils.data.DataLoader(scorehmr_kps2d_dataset,
                                                                            batch_size=boxes.shape[0], shuffle=False,
                                                                            num_workers=0)
                    scorehmr_batch = next(iter(scorehmr_kps2d_dataloader))
                    scorehmr_kps2d_img = scorehmr_batch['img']  # (1, 3, 244, 244)
                    scorehmr_kps2d_personid = scorehmr_batch['personid']  # (1)
                    scorehmr_kps2d_boxcenter = scorehmr_batch['box_center']  # (1, 2)
                    scorehmr_kps2d_imgsize = scorehmr_batch['img_size']  # (1, 2)
                    scorehmr_kps2d = scorehmr_batch['keypoints_2d']  # (1, 63, 3)
                    scorehmr_kps2d_img_view.append(scorehmr_kps2d_img[0])
                    scorehmr_kps2d_personid_view.append(scorehmr_kps2d_personid[0])
                    scorehmr_kps2d_boxcenter_view.append(scorehmr_kps2d_boxcenter[0])
                    scorehmr_kps2d_imgsize_view.append(scorehmr_kps2d_imgsize[0])
                    scorehmr_kps2d_view.append(scorehmr_kps2d[0])
                if hmr2_out is None:
                    self.filelist.remove(filename)
                    filename = self.filelist[idx]
                    # variable in keypoints2d scorehmr
                    pred_cam_t_view = []
                    center_view = []
                    img_size_view = []
                    scorehmr_kps2d_img_view = []  # (1, 3, 244, 244)
                    scorehmr_kps2d_personid_view = []  # (1)
                    scorehmr_kps2d_boxcenter_view = []  # (1, 2)
                    scorehmr_kps2d_imgsize_view = []  # (1, 2)
                    scorehmr_kps2d_view = []  # (1, 63, 3)
                    # variable in multiview scorehmr
                    keypoints2d_view = []
                    img_patch = []
                    pred_pose_view = []
                    scorehmr_pred_betas_view = []
                    # variable in render mesh
                    all_verts_view = []
                    all_cam_t_view = []
                    scaled_focal_length_view = []
                    hmr2_img_size_view = []
                    img_cv2_view = []
                    imgname_view = []
                    pred_betas_view = []
                    global_orient_view = []
                    body_pose_view = []
                    break
                all_verts_view.append(all_verts)
                all_cam_t_view.append(all_cam_t)
                # variable in multiview scorehmr
                keypoints_3d = np.zeros((44, 4)).astype(np.float32)
                scorehmr_body_pose = np.zeros(72, dtype="float32")
                scorehmr_betas = np.zeros(10, dtype="float32")
                scorehmr_has_body_pose = 0
                scorehmr_has_betas = 0
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                smpl_params_n = {
                    "global_orient": scorehmr_body_pose[:3],
                    "body_pose": scorehmr_body_pose[3:],
                    "betas": scorehmr_betas,
                }
                has_smpl_params_n = {
                    "global_orient": scorehmr_has_body_pose,
                    "body_pose": scorehmr_has_body_pose,
                    "betas": scorehmr_has_betas,
                }
                img_patch_n, keypoints_2d_n, keypoints_3d_n, smpl_params_n, has_smpl_params_n, img_size_n = get_example(
                    img_path,
                    center_x,  # bbox center x in img
                    center_y,  # box center y in img
                    bbox_width,  # bbox width
                    bbox_height,  # bbox height
                    keypoints_2d,  # (N, 3), keypoints2d in img
                    keypoints_3d,  # (N, 4)
                    smpl_params_n,  # SMPL annotation
                    has_smpl_params_n,
                    flip_kp_permutation=self.flip_keypoint_permutation,
                    patch_width=self.img_size,  # Output box width
                    patch_height=self.img_size,  # Output box height
                    mean=self.mean,  # (3,) normalizing the input image
                    std=self.std,  # (3,) normalizing the input image
                    do_augment=False,
                    augm_config=None,
                    load_image=True,
                )
                img_patch.append(img_patch_n)
                pred_pose_n = torch.cat(
                    (hmr2_out['pred_smpl_params']['global_orient'], hmr2_out['pred_smpl_params']['body_pose']), axis=1)
                pred_pose_n = pred_pose_n[idx].detach().cpu().numpy()
                pred_pose_view.append(pred_pose_n)
                pred_betas_n = hmr2_out['pred_smpl_params']['betas'][idx].detach().cpu().numpy()
                scorehmr_pred_betas_view.append(pred_betas_n)
                center_view.append(center)


        item = {}
        # multiview scorehmr
        keypoints2d_view = np.stack(keypoints2d_view, axis=0).astype(np.float32)
        item["keypoints_2d"] = torch.from_numpy(keypoints2d_view[..., :-1])
        img_patch = np.stack(img_patch, axis=0)
        item["multiview_img"] = torch.from_numpy(img_patch)
        pred_pose_view = np.stack(pred_pose_view, axis=0).astype(np.float32)
        item["pred_pose"] = torch.from_numpy(pred_pose_view)  # keypoints2d_fitting
        scorehmr_pred_betas_view = np.stack(scorehmr_pred_betas_view, axis=0).astype(np.float32)
        item["pred_betas"] = torch.from_numpy(scorehmr_pred_betas_view)  # scorehmr_pred_betas  keypoints2d_fitting
        # 2d keypoints scorehmr
        scorehmr_kps2d_img_view = np.stack(scorehmr_kps2d_img_view, axis=0).astype(np.float32)
        item["img"] = torch.from_numpy(scorehmr_kps2d_img_view)
        scorehmr_kps2d_personid_view = np.stack(scorehmr_kps2d_personid_view, axis=0).astype(np.float32)
        item["personid"] = torch.from_numpy(scorehmr_kps2d_personid_view)
        scorehmr_kps2d_boxcenter_view = np.stack(scorehmr_kps2d_boxcenter_view, axis=0).astype(np.float32)
        item["box_center"] = torch.from_numpy(scorehmr_kps2d_boxcenter_view)
        scorehmr_kps2d_imgsize_view = np.stack(scorehmr_kps2d_imgsize_view, axis=0).astype(np.float32)
        item["img_size"] = torch.from_numpy(scorehmr_kps2d_imgsize_view)
        scorehmr_kps2d_view = np.stack(scorehmr_kps2d_view, axis=0).astype(np.float32)
        item["joints_conf"] = torch.from_numpy(scorehmr_kps2d_view[..., -1]).unsqueeze(2)
        pred_cam_t_view = np.stack(pred_cam_t_view, axis=0).astype(np.float32)
        item["init_cam_t"] = torch.from_numpy(pred_cam_t_view)
        item["joints_2d"] = torch.from_numpy(scorehmr_kps2d_view[..., :-1])
        item["camera_center"] = item["img_size"] / 2
        # 4dhumans
        item["imgname"] = imgname_view
        item["scaled_focal_length"] = scaled_focal_length_view
        item["all_verts"] = all_verts_view
        item["all_cam_t"] = all_cam_t_view
        item["hmr2_img_size"] = hmr2_img_size_view
        item["img_cv2"] = img_cv2_view
        item["focal_length"] = 5000 * torch.ones(8, 2, dtype=item["joints_2d"].dtype)
        item["global_orient"] = global_orient_view
        item["body_pose"] = body_pose_view
        return item

def worker(args, rank, subset_mname):
    rank = rank % 8
    device = torch.device(f"cuda:{rank}")
    # load 4dhumans model
    hmr2_model, model_cfg_hmr2 = load_hmr2(args.hmr2_checkpoint)
    hmr2_model = hmr2_model.to(device)
    hmr2_model.eval()

    # load a diffusion-based model score_hmr used multiview guidance
    name = "score_hmr"
    predictions = "hmr2"
    optim_iters = 2
    use_default_ckpt = True
    sample_start = 100
    ddim_step_size = 10
    model_cfg = model_config()
    model_cfg.defrost()
    model_cfg.EXTRA.LOAD_PREDICTIONS = predictions
    model_cfg.GUIDANCE.OPTIM_ITERS = optim_iters
    model_cfg.freeze()
    extra_args = {
        "name": name,
        "multiview_guidance": True,
        "use_default_ckpt": use_default_ckpt,
        "optim_iters": optim_iters,
        "sample_start": sample_start,
        "ddim_step_size": ddim_step_size,
        "device": device
    }
    diffusion_model = load_diffusion_model(model_cfg, **extra_args)

    # load a diffusion-based scorehmr used 2d keypoints guidance
    keypoints2d_extra_args = {
        "keypoint_guidance": True,
        "use_default_ckpt": True,
        "device": device,
    }
    keypoints2d_model_cfg = model_config()
    keypoints2d_model_cfg.defrost()
    keypoints2d_model_cfg.GUIDANCE.OPTIM_ITERS = 10
    keypoints2d_model_cfg.freeze()
    keypoints2d_diffusion_model = load_diffusion_model(keypoints2d_model_cfg, **keypoints2d_extra_args)

    # load a image processed model pare
    multiview_pare = load_pare(model_cfg.SMPL).to(device)  # for multiview scorehmr
    multiview_pare.eval()
    mutliview_img_feat_standarizer = StandarizeImageFeatures(
        backbone=model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
        use_betas=False,
        device=device,
    )

    kps2d_pare = load_pare(keypoints2d_model_cfg.SMPL).to(device)  # for 2d keypoints scorehmr
    kps2d_pare.eval()
    kps2d_img_feat_standarizer = StandarizeImageFeatures(
        backbone=keypoints2d_model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
        use_betas=False,
        device=device,
    )

    # load renderer
    renderer = Renderer(model_cfg_hmr2, faces=hmr2_model.smpl.faces)
    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=5000.,
    )

    # load detector
    cfg_path = Path(hmr2.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Dataset
    dataset = BatchedImageDataset(args.kps2d_folder, args.img_folder, subset_mname, detector, model_cfg_hmr2, hmr2_model, device)
    for j, batch in enumerate(tqdm(dataset)):
        # Run iterative refinement with ScoreHMR(multiview)
        batch = recursive_to(batch, device)
        batch_size = batch["keypoints_2d"].size(0)
        with torch.no_grad():
            pare_out = multiview_pare(batch["multiview_img"], get_feats=True)
        cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
        cond_feats = mutliview_img_feat_standarizer(cond_feats)
        dm_out = None
        with torch.no_grad():
            dm_out = diffusion_model.sample(batch, cond_feats, batch_size=batch_size)
        if dm_out is not None:
            multiview_pred_smpl_params = prepare_smpl_params(dm_out['x_0'], num_samples=1, use_betas=False,
                                                             pred_betas=batch['pred_betas'])
            global_orient = multiview_pred_smpl_params['global_orient']
            body_pose = multiview_pred_smpl_params['body_pose']
            pred_pose = torch.cat((global_orient, body_pose), axis=1)
            batch["pred_pose"] = pred_pose
            pred_betas = multiview_pred_smpl_params['betas']
            batch["betas"] = pred_betas
        else:
            batch["betas"] = batch["pred_betas"]
        # Run iterative refinement with ScoreHMR(2d keypoints)
        with torch.no_grad():
            pare_out = kps2d_pare(batch["img"], get_feats=True)
        cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
        cond_feats = kps2d_img_feat_standarizer(cond_feats)  # normalize image features
        with torch.no_grad():
            dm_out = keypoints2d_diffusion_model.sample(batch, cond_feats, batch_size=batch_size)
        kps2d_pred_smpl_params = prepare_smpl_params(dm_out['x_0'], num_samples=1, use_betas=False,
                                                     pred_betas=batch['pred_betas'])
        # render pose in image
        imgname_view = batch["imgname"]
        scaled_focal_length_view = batch["scaled_focal_length"]
        all_cam_t_view = batch["all_cam_t"]  # camera parameters predicted by hmr2
        hmr2_img_size_view = batch["hmr2_img_size"]
        img_cv2_view = batch["img_cv2"]
        global_orient_view = batch["global_orient"]
        body_pose_view = batch["body_pose"]
        pred_betas_view = batch["pred_betas"]
        for i in range(len(imgname_view)):
            imgname = imgname_view[i]
            scaled_focal_length = scaled_focal_length_view[i]
            global_orient = global_orient_view[i].unsqueeze(0)
            body_pose = body_pose_view[i].unsqueeze(0)
            pred_betas = pred_betas_view[i].unsqueeze(0)
            all_cam_t = all_cam_t_view[i]
            hmr2_img_size = hmr2_img_size_view[i]
            img_cv2 = img_cv2_view[i]
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            human4d_pred_smpl_params = {
                "global_orient": global_orient,  # (batch, 1, 3, 3)
                "body_pose": body_pose,  # (batch, 23, 3, 3)
                "betas": pred_betas  # (batch, 10)
            }
            img_path = args.img_folder + "/" + imgname + ".jpg"
            img = cv2.imread(img_path)
            save_img_path = args.save_render_dir + "/" + imgname + ".png"
            cv2.imwrite(save_img_path, img)
            if args.human4d_render:
                smpl_out = diffusion_model.smpl(**human4d_pred_smpl_params, pose2rot=False)
                verts = smpl_out.vertices.cpu().numpy()
                all_verts = [verts[0]]
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=hmr2_img_size,
                                                         **misc_args)
                # Overlay image
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :,
                                                                                                          3:]
                render_img_path = args.save_render_dir + "/" + imgname + "_human4d_render.png"
                cv2.imwrite(render_img_path, 255 * input_img_overlay[:, :, ::-1])
                obj_save_path = args.save_render_dir + "/" + imgname + "_human4d.obj"
                tmesh = renderer.vertices_to_trimesh(verts[0], all_cam_t[0], LIGHT_BLUE)
                tmesh.export(obj_save_path)
            if args.multiview_render:
                smpl_out = diffusion_model.smpl(**multiview_pred_smpl_params, pose2rot=False)
                verts = smpl_out.vertices.cpu().numpy()
                all_verts = [verts[i]]
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=hmr2_img_size,
                                                         **misc_args)
                # Overlay image
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :,
                                                                                                          3:]
                render_img_path = args.save_render_dir + "/" + imgname + "_multiview_render.png"
                cv2.imwrite(render_img_path, 255 * input_img_overlay[:, :, ::-1])
                obj_save_path = args.save_render_dir + "/" + imgname + "_multiview.obj"
                tmesh = renderer.vertices_to_trimesh(verts[i], all_cam_t[0], LIGHT_BLUE)
                tmesh.export(obj_save_path)
            if args.kps2d_render:
                smpl_out = diffusion_model.smpl(**kps2d_pred_smpl_params, pose2rot=False)
                verts = smpl_out.vertices.cpu().numpy()
                all_verts = [verts[i]]
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=hmr2_img_size,
                                                         **misc_args)
                # Overlay image
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :,
                                                                                                          3:]
                render_img_path = args.save_render_dir + "/" + imgname + "_kps2d_render.png"
                cv2.imwrite(render_img_path, 255 * input_img_overlay[:, :, ::-1])
                obj_save_path = args.save_render_dir + "/" + imgname + "_kps2d.obj"
                tmesh = renderer.vertices_to_trimesh(verts[i], all_cam_t[0], LIGHT_BLUE)
                tmesh.export(obj_save_path)
            # save optimized motion
            motion_path = args.save_motion_dir + "/" + imgname + ".npy"
            global_orient = kps2d_pred_smpl_params["global_orient"].detach().cpu().numpy()  # (batch, 1, 3, 3)
            body_pose = kps2d_pred_smpl_params["body_pose"].detach().cpu().numpy()  # (batch, 23, 3, 3)
            betas = kps2d_pred_smpl_params["betas"].detach().cpu().numpy()  # (batch, 10)
            global_orient = global_orient[i]
            body_pose = body_pose[i]
            betas = betas[i]
            smpl_params = {
                "global_orient": global_orient,
                "body_pose": body_pose,
                "betas": betas
            }
            np.save(motion_path, [smpl_params])


def distribute_data_over_processes(motion_list, num_processes):
    avg = len(motion_list) // num_processes
    remainder = len(motion_list) % num_processes

    subsets = []
    start_idx = 0
    for _ in range(num_processes):
        subset_size = avg + (1 if remainder > 0 else 0)
        subset = motion_list[start_idx : start_idx + subset_size]
        subsets.append(subset)
        start_idx += subset_size
        remainder -= 1
    return subsets

def main(args, process_num=1):
    exist_motion_list = glob.glob(args.save_motion_dir + "/*_c01.npy")
    all_jpgs_files = sorted(os.listdir(args.img_folder))
    print("all jpg num: ", len(all_jpgs_files))
    error_motions = ["20221002_8cee0a0503_subj20_000011", "20221002_8cee0a0503_subj20_000012", "20220813_8f66816103_subj22_000117",
                     "20220928_c99d182f05_subj17_003605", "20221002_8cee0a0502_subj20_003437", "20220927_b533ab8e04_subj30_003612"]
    if os.path.exists(args.motion_list_pkl):
        with open(args.motion_list_pkl, 'rb') as file:
            motion_list = pickle.load(file)
    else:
        motion_list = []
        i = 0
        length = len(all_jpgs_files)
        while i < length:
            print(f"process:{i}/{length}")
            f = all_jpgs_files[i]  # 20220816_ab556d6101_subj07_000235_c05.jpg
            imgname = f.split(".")[0]  # 20220816_ab556d6101_subj07_000235_c05
            mname = imgname + ".npy"
            f_idx = int(f.split("_")[3])  # remove the first 10 frames in video to ignore image without human
            if mname in exist_motion_list or f_idx <= 10:
                i = i + 8
                continue
            f_c8 = all_jpgs_files[i + 7]  # each pose has 8 images from different views
            sub_names = imgname.split("_")
            filename = "_".join(sub_names[0:-1])
            if filename not in f_c8 or filename in error_motions:
                i = i + 1
                continue
            next_f = all_jpgs_files[min(i + 8 * 8, length-1)]  # select 1 from 8 frames
            next_motion = "_".join(next_f.split("_")[:3])
            if next_motion in f:   # next_frame(next_f) and frame(f) from same video
                i = i + 8 * 8
            else:  # next_frame(next_f) and frame(f) from different video
                i = i + 8
            motion_list.append(filename)
        with open(args.motion_list_pkl, 'wb') as file:
            pickle.dump(motion_list, file)
    print("len(motion_list):", len(motion_list))
    subsets = distribute_data_over_processes(motion_list, process_num)
    processes = []
    for rank, subset in enumerate(subsets):
        p = mp.Process(target=worker, args=(args, rank, subset))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FreeMan motion annotation pipeline')
    parser.add_argument('--hmr2_checkpoint', type=str, default='/data/jiaqi_liao/4D-Humans/logs/train/runs/hmr_hand_and_foot_4Dhumans/checkpoints/epoch=25-step=85000.ckpt', help='Path to pretrained hmr2 model checkpoint')
    parser.add_argument('--img_folder', type=str, default='/data/boning_zhang/Freeman/handdataset/images', help='Folder with input images')
    parser.add_argument('--kps2d_folder', type=str, default='/data/boning_zhang/Freeman/keypoints2d_foot', help='Folder with 2d keypoints')
    parser.add_argument('--save_motion_dir', type=str, default='/data/boning_zhang/Freeman/motion_foot_62', help='Folder to save optimized motion')
    parser.add_argument('--save_render_dir', type=str, default='/data/boning_zhang/Freeman/scorehmr', help='Folder to save render motion')
    parser.add_argument('--human4d_render', dest='human4d_render', action='store_true', default=True, help='If set, render pose predicted by 4dhumans')
    parser.add_argument('--multiview_render', dest='multiview_render', action='store_true', default=True, help='If set, render pose optimized by multiview guidance')
    parser.add_argument('--kps2d_render', dest='kps2d_render', action='store_true', default=True, help='If set, render pose optimized by multiview guidance and keypoints2d guidance')
    parser.add_argument('--motion_list_pkl', type=str, default='/data/boning_zhang/Freeman/motion.pkl', help='Cache image filename to be processed to save time')
    args = parser.parse_args()

    if not os.path.exists(args.save_motion_dir):
        os.makedirs(args.save_motion_dir, exist_ok=True)
    if not os.path.exists(args.save_render_dir):
        os.makedirs(args.save_render_dir, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')
    process_num = 2
    main(args, process_num=process_num)


