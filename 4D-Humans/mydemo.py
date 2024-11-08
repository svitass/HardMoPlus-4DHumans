# -*- coding: utf-8 -*-
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import pdb
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

import pickle
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='../HandBody-fusion/example_data', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='../HandBody-fusion/demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--save_smpl', dest='save_smpl', action='store_true', default=False, help='If set, save smpl para')
    parser.add_argument('--save_render', dest='save_render', action='store_true', default=False, help='If set, save render pic')
    parser.add_argument('--cpu', dest='cpu', action='store_true', default=False, help='If set, use cpu')
    parser.add_argument('--notonlyone',dest='notonlyone', action='store_true', default=False, help='If set, use cpu')
    args = parser.parse_args()

   

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # def get_smplx_faces(v2020=True):
    #     SMPL_MODEL_DIR="/data/boning_zhang/PyMAF-X/data/smpl"
    #     import sys
    #     sys.path.append('/data/boning_zhang/PyMAF-X/models')
    #     sys.path.append('/data/boning_zhang/PyMAF-X')
    #     # 导入目标模块
    #     from smpl import SMPLX
    #     smplx = SMPLX(os.path.join(SMPL_MODEL_DIR, 'SMPLX_NEUTRAL_2020.npz'), batch_size=1)
    #     return smplx.faces
    # faces = get_smplx_faces() #要是array n*3的
    # Setup the renderer
    #renderer = Renderer(model_cfg,faces)
    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)
    import joblib
    data =joblib.load("../HandBody-fusion/output.pkl") 
    #pred_vertices =data['smplx_verts']
    pred_vertices =data['verts']
    # pdb.set_trace()
    image_dir ="../HandBody-fusion/example_vis"         
    # 获取目录中所有文件的列表
    file_list = os.listdir(image_dir)

    # 过滤出所有图片文件（假设图片格式为 jpg, png 等）
    image_files = sorted([f for f in file_list if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif'))])
    # Iterate over all images in folder
    for n, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        img_cv2 = cv2.imread(str(image_path))

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        if not args.notonlyone:
            boxes = det_instances.pred_boxes.tensor[valid_idx]
            if len(boxes) > 0:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                max_area_index = areas.argmax()
                valid_idx = valid_idx.cpu()
                new_valid_idx = np.zeros_like(valid_idx, dtype=bool)
                new_valid_idx[np.where(valid_idx)[0][max_area_index]] = True
                boxes=det_instances.pred_boxes.tensor[new_valid_idx].cpu().numpy()
            else:
                boxes=boxes.cpu().numpy()
        else:
            boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        # 这里把bbox crop出来并且归一化了，图片被归一化
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            # Render the result
            batch_size = 1
            
            for m in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(image_path))
                person_id = int(batch['personid'][m])
                white_img = (torch.ones_like(batch['img'][m]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][m].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()
                regression_img = renderer(pred_vertices[n],
                                        out['pred_cam_t'][m].detach().cpu().numpy(),
                                        batch['img'][m],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                final_img = np.concatenate([input_patch, regression_img], axis=1)

                if args.side_view:
                    side_img = renderer(pred_vertices[n],
                                            out['pred_cam_t'][m].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([final_img, side_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = pred_vertices[n]
                cam_t = pred_cam_t_full[m]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))
                

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[m], **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

        end = time.time()
        print(end - start)

        camera_translation = pred_cam_t_full.copy()
        tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
        tmesh.export(os.path.join(args.out_folder, f'{img_fn}.obj'))


if __name__ == '__main__':
    main()
