import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))

import os
import pickle
import cv2
import torch
import argparse
import shutil
import numpy as np
from pathlib import Path
import warnings
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.utils.renderer import Renderer, cam_crop_to_full
import multiprocessing as mp
from score_hmr.utils import *
from score_hmr.configs import model_config
from score_hmr.datasets.videt_dataset import ViTDetDataset
from tqdm import tqdm
from score_hmr.models.model_utils import load_diffusion_model, load_pare
from score_hmr.utils.rotation_convert import matrix_to_axis_angle, axis_angle_to_matrix
warnings.filterwarnings('ignore')

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def worker(args, rank, img_subset, pyd_subset, save_subset):
    rank = rank % 8
    device = torch.device(f"cuda:{rank}")

    # Download and load checkpoints.
    download_models(CACHE_DIR_4DHUMANS)
    # Copy SMPL model to the appropriate path for HMR 2.0 if it does not exist.
    if not os.path.isfile(f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl'):
        shutil.copy('data/smpl/SMPL_NEUTRAL.pkl', f'{CACHE_DIR_4DHUMANS}/data/smpl/')
    hmr2_model, model_cfg_hmr2 = load_hmr2(args.hmr2_checkpoint)

    # Setup HMR2.0 model.
    hmr2_model = hmr2_model.to(device)
    hmr2_model.eval()

    # Load human detector.
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg_hmr2, faces=hmr2_model.smpl.faces)
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=5000.,
    )

    # Load config.
    model_cfg = model_config()
    model_cfg.defrost()
    model_cfg.GUIDANCE.OPTIM_ITERS = 10
    model_cfg.freeze()

    # Load PARE model.
    pare = load_pare(model_cfg.SMPL).to(device)
    pare.eval()

    img_feat_standarizer = StandarizeImageFeatures(
        backbone=model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
        use_betas=False,
        device=device,
    )

    # Load diffusion model.
    extra_args = {
        "keypoint_guidance": True,
        "use_default_ckpt": True,
        "device": device,
    }
    diffusion_model = load_diffusion_model(model_cfg, **extra_args)

    # Iterate over all images in folder.
    for img_path, pyd_path, save_path in tqdm(zip(img_subset, pyd_subset, save_subset)):
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image.
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # load initial annotation of image
        with open(pyd_path, 'rb') as f:
            data = pickle.load(f)
            keypoints_2d = data["keypoints_2d"]  # (44, 3)
            body_pose = data["body_pose"]  # (72,)
            init_body_pose = torch.tensor(body_pose.reshape(24, 3))
            # wrist: 20, 21  hand: 22, 23
            init_body_pose = axis_angle_to_matrix(init_body_pose)  # (24, 3, 3)
        body_keypoints_2d = np.array([keypoints_2d[:25]])  # (1, 25, 3)

        # get person bbox id
        idx = 0
        max_iou = -float('inf')
        x_min = np.min(body_keypoints_2d[0, :, 0])
        y_min = np.min(body_keypoints_2d[0, :, 1])
        x_max = np.max(body_keypoints_2d[0, :, 0])
        y_max = np.max(body_keypoints_2d[0, :, 1])
        for i, (hmr2_x_min, hmr2_y_min, hmr2_x_max, hmr2_y_max) in enumerate(pred_bboxes):
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
            iou = area_overlap / area_union if area_union > 0 else 0
            if iou > max_iou:
                max_iou = iou
                idx = i
        if max_iou != (-float('inf')):
            pred_bboxes = np.array([pred_bboxes[idx]])
        else:
            continue

        # Create separate dataset of HMR 2.0 and ScoreHMR, since the input should be of different resolution.
        dataset_hmr2 = ViTDetDataset(model_cfg_hmr2, img_cv2, pred_bboxes)  # pred_bboxes: (10, 4)
        dataloader_hmr2 = torch.utils.data.DataLoader(dataset_hmr2, batch_size=pred_bboxes.shape[0], shuffle=False,
                                                      num_workers=0)

        dataset = ViTDetDataset(model_cfg_hmr2, img_cv2, pred_bboxes, body_keypoints_2d, is_hmr2=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=pred_bboxes.shape[0], shuffle=False, num_workers=0)

        # Get predictions from HMR 2.0
        hmr2_batch = recursive_to(next(iter(dataloader_hmr2)), device)
        with torch.no_grad():
            out = hmr2_model(hmr2_batch)  # image -> SMPL

        pred_cam = out['pred_cam']
        batch_size = pred_cam.shape[0]
        box_center = hmr2_batch["box_center"].float()
        box_size = hmr2_batch["box_size"].float()
        img_size = hmr2_batch["img_size"].float()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size)

        # Run iterative refinement with ScoreHMR.
        batch = recursive_to(next(iter(dataloader)), device)
        with torch.no_grad():
            pare_out = pare(batch["img"], get_feats=True)
        cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
        cond_feats = img_feat_standarizer(cond_feats)  # normalize image features

        # Prepare things for model fitting.
        batch["camera_center"] = batch["img_size"] / 2
        batch["joints_2d"] = batch["keypoints_2d"][:, :, :2]
        batch["joints_conf"] = batch["keypoints_2d"][:, :, [2]]
        batch["focal_length"] = model_cfg.EXTRA.FOCAL_LENGTH * torch.ones(
            batch_size,
            2,
            device=device,
            dtype=batch["keypoints_2d"].dtype,
        )
        batch['pred_betas'] = out['pred_smpl_params']['betas']
        batch['pred_pose'] = torch.cat((out['pred_smpl_params']['global_orient'], out['pred_smpl_params']['body_pose']),
                                       axis=1)
        batch["init_cam_t"] = pred_cam_t_full

        # Run ScoreHMR.
        print(f'=> Running ScoreHMR for image: {img_fn}')
        with torch.no_grad():
            dm_out = diffusion_model.sample(
                batch, cond_feats, batch_size=batch_size
            )

        pred_smpl_params = prepare_smpl_params(
            dm_out['x_0'],
            num_samples=1,
            use_betas=False,
            pred_betas=batch["pred_betas"],
        )

        # # replace wrist in optimized pose  wrist_index(20, 21) - 1(root)  hand_index:(22, 23) - 1(root)
        # pred_smpl_params["body_pose"][:, 19, :] = init_body_pose[20]
        # pred_smpl_params["body_pose"][:, 20, :] = init_body_pose[21]
        # pred_smpl_params["body_pose"][:, 21, :] = init_body_pose[22]
        # pred_smpl_params["body_pose"][:, 22, :] = init_body_pose[23]

        # save new annotation data as pyd file
        sname = os.path.basename(save_path)
        save_img_path = args.save_motion_dir + "/" + sname + ".jpg"
        shutil.copy(img_path, save_img_path)
        save_pyd_path = args.save_motion_dir + "/" + sname + ".data.pyd"
        global_orient = matrix_to_axis_angle(pred_smpl_params['global_orient'].detach().cpu()).numpy()
        body_pose = matrix_to_axis_angle(pred_smpl_params['body_pose'].detach().cpu()).numpy()
        poses = np.concatenate((global_orient, body_pose), axis=-2).flatten()
        body_pose = poses
        betas = pred_smpl_params['betas'].detach().cpu().numpy()
        data["body_pose"] = body_pose.astype(np.float32)
        data["betas"] = betas.astype(np.float32)
        data["has_body_pose"] = 1
        data["has_betas"] = 1
        with open(save_pyd_path, "wb") as f:
            pickle.dump([data], f)

        smpl_out = diffusion_model.smpl(**pred_smpl_params, pose2rot=False)
        opt_verts = smpl_out.vertices.cpu().numpy()
        opt_cam_t = dm_out['camera_translation'].cpu().numpy()
        
        # Render front view.
        if args.motion_render:
            print(f'=> Rendering image: {img_fn}')
            render_res = img_size[0].cpu().numpy()
            cam_view = renderer.render_rgba_multiple(opt_verts, cam_t=opt_cam_t, render_res=render_res, **misc_args)
            # Overlay and save image.
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])],
                                       axis=2)  # Add alpha channel  # (720, 1280, 4)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            cv2.imwrite(args.save_render_dir + f'/opt_{img_fn}.png', 255 * input_img_overlay[:, :, ::-1])

def distribute_data_over_processes(img_list, pyd_list, save_list, process_num):
    avg = len(img_list) // process_num
    remainder = len(img_list) % process_num
    img_subsets = []
    pyd_subsets = []
    save_subsets = []
    start_idx = 0
    for _ in range(process_num):
        subset_size = avg + (1 if remainder > 0 else 0)
        img_subset = img_list[start_idx : start_idx + subset_size]
        pyd_subset = pyd_list[start_idx : start_idx + subset_size]
        save_subset = save_list[start_idx : start_idx + subset_size]
        img_subsets.append(img_subset)
        pyd_subsets.append(pyd_subset)
        save_subsets.append(save_subset)
        start_idx += subset_size
        remainder -= 1
    return img_subsets, pyd_subsets, save_subsets

def main(args, process_num=1):
    img_list = []
    pyd_list = []
    save_list = []  # save optimized motion path
    pyd_10000000_list = ["changquan", "bagua", "bajiquan", "daoshu", "jianshu", "qiangshu", "rumba", "taijiquan", "xingyi"]
    for img_data_root, pyd_data_root in zip(args.img_folder, args.pyd_folder):
        motion_classes = os.listdir(img_data_root)
        print("load motion...")
        print("motion_classes:", motion_classes)
        for mclass in tqdm(motion_classes):
            if 'tango' not in mclass:
                continue
            print(f"load motion {mclass}")
            img_mclass_path = img_data_root + "/" + mclass
            class_id_list = os.listdir(img_mclass_path)
            pyd_mclass_path = pyd_data_root + "/" + mclass
            for class_id in tqdm(class_id_list):
                img_dir = img_mclass_path + "/" + class_id
                imgname_list = os.listdir(img_dir)
                pyd_dir = pyd_mclass_path + "/" + class_id
                for imgname in tqdm(imgname_list[::10]):  # pydname: 000001_10000000.pyd
                    img_path = img_dir + "/" + imgname  # 000001.jpeg
                    if os.path.basename(img_data_root) == "dance_data_renamed_image_rename" or mclass in pyd_10000000_list:  # dance data
                        pydname = imgname.split(".")[0] + "_10000000.pyd"
                    else:
                        pydname = imgname.split(".")[0] + "_1.pyd"
                    pyd_path = pyd_dir + "/" + pydname
                    if os.path.exists(img_path) and os.path.exists(pyd_path):
                        save_path = args.save_motion_dir + "/" + mclass + "_" + class_id + "_" + imgname.split(".")[0]
                        sname = os.path.basename(save_path)
                        save_pyd_path = args.save_motion_dir + "/" + sname + ".data.pyd"
                        if os.path.exists(save_pyd_path):
                            continue
                        else:
                            img_list.append(img_path)
                            pyd_list.append(pyd_path)
                            save_list.append(save_path)
    length = len(img_list)
    print("img_list length:", length)

    img_subsets, pyd_subsets, save_subsets = distribute_data_over_processes(img_list, pyd_list, save_list, process_num)
    processes = []
    for rank, (img_subset, pyd_subset, save_subset) in enumerate(zip(img_subsets, pyd_subsets, save_subsets)):
        p = mp.Process(target=worker, args=(args, rank, img_subset, pyd_subset, save_subset))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FreeMan motion annotation pipeline')
    parser.add_argument('--hmr2_checkpoint', type=str,
                        default='/data/jiaqi_liao/4D-Humans/logs/train/runs/hmr_hand_and_foot_4Dhumans/checkpoints/epoch=25-step=85000.ckpt',
                        help='Path to pretrained hmr2 model checkpoint')
    parser.add_argument('--img_folder', type=list, default=["/data/jiaqi_liao/dance_data_renamed_image_rename"], help='Folder with input images')
    # pyd_data_roots = ["/data/jiaqi_liao/dance_pyd_source", "/data/jiaqi_liao/kungfu_data_rename_image"]
    parser.add_argument('--pyd_folder', type=list, default=["/data/jiaqi_liao/dance_pyd_source"], help='Folder with input data annotation')
    parser.add_argument('--save_motion_dir', type=str, default='/data/boning_zhang/Freeman/hardmo_all_62', help='Folder to save optimized motion')
    parser.add_argument('--save_render_dir', type=str, default='/data/boning_zhang/Freeman/scorehmr', help='Folder to save render motion')
    parser.add_argument('--motion_render', dest='motion_render', action='store_true', default=True, help='If set, render optimized pose')
    args = parser.parse_args()

    if not os.path.exists(args.save_motion_dir):
        os.makedirs(args.save_motion_dir, exist_ok=True)
    if not os.path.exists(args.save_render_dir):
        os.makedirs(args.save_render_dir, exist_ok=True)


    torch.multiprocessing.set_start_method('spawn')
    process_num = 1
    main(args, process_num=process_num)
