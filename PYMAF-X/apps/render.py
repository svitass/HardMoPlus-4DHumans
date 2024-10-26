# -*- coding: utf-8 -*-
# This script is borrowed and extended from https://github.com/mkocabas/VIBE/blob/master/demo.py and https://github.com/nkolot/SPIN/blob/master/demo.py
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from copyreg import pickle
import enum
import os
import copy
#os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
import pickle as pkle

import pdb
import cv2
import time
import json
import shutil
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os.path as osp
from matplotlib.image import imsave
from skimage.transform import resize
from torchvision.transforms import Normalize
from collections import OrderedDict

from core.cfgs import cfg, parse_args
from models import blend, hmr, pymaf_net
from models.smpl import get_partial_smpl, SMPL, SMPLX
from core import path_config, constants
from datasets.inference import Inference
from utils.renderer import PyRenderer
from utils.imutils import crop
from utils.demo_utils import (
    download_url,
    convert_crop_cam_to_orig_img,
    video_to_images,
    images_to_video,
)
from utils.geometry import convert_to_full_img_cam
from models.getout import get_out


from os.path import join, expanduser


MIN_NUM_FRAMES = 1

def prepare_rendering_results(person_data, nframes):
    frame_results = [{} for _ in range(nframes)]
    for idx, frame_id in enumerate(person_data['frame_ids']):
        person_id = person_data['person_ids'][idx],
        frame_results[frame_id][person_id] = {
            'verts': person_data['verts'][idx],
            'smplx_verts': person_data['smplx_verts'][idx] if 'smplx_verts' in person_data else None,
            'cam': person_data['orig_cam'][idx],
            'cam_t': person_data['orig_cam_t'][idx] if 'orig_cam_t' in person_data else None,
            # 'cam': person_data['pred_cam'][idx],
        }

    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results


def run_demo(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.image_folder is None:
        video_file = args.vid_file

        # ========= [Optional] download the youtube video ========= #
        if video_file.startswith('https://www.youtube.com'):
            print(f'Donwloading YouTube video \"{video_file}\"')
            video_file = download_url(video_file, '/tmp')

            if video_file is None:
                exit('Youtube url is not valid!')

            print(f'YouTube Video has been downloaded to {video_file}...')

        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')
        
        output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))

        image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
    else:
        image_folder = args.image_folder
        num_frames = len(os.listdir(image_folder))
        img_shape = cv2.imread(osp.join(image_folder, os.listdir(image_folder)[0])).shape

        output_path = os.path.join(args.output_folder, osp.split(image_folder)[-1])

    os.makedirs(output_path, exist_ok=True)

    print(f'Input video number of frames {num_frames}')


    args.device = device
    args.pin_memory = True if torch.cuda.is_available() else False




    smpl2limb_vert_faces = get_partial_smpl(args.render_model) #smplx

    smpl2part = smpl2limb_vert_faces[args.render_part]['vids']
    part_faces = smpl2limb_vert_faces[args.render_part]['faces']


    pred_results = joblib.load("/data/boning_zhang/PyMAF-X/output/examples/output.pkl")

    if not args.no_render:
      
        # ========= Render results as a single video ========= #
        
        renderer = PyRenderer(vis_ratio=args.render_vis_ratio)

        output_img_folder = os.path.join(output_path, osp.split(image_folder)[-1] + '_output')
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_img_folder + '/arm', exist_ok=True)

        print(f'Rendering results, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(pred_results, num_frames)

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        color_type = 'purple'

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if args.render_ratio != 1:
                img = resize(img, (int(img.shape[0] * args.render_ratio), int(img.shape[1] * args.render_ratio)), anti_aliasing=True)
                img = (img * 255).astype(np.uint8)

            raw_img = img.copy()

            img_full = img.copy()
            img_arm = img.copy()

            if args.empty_bg:
                empty_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                if args.render_model == 'smplx':
                    frame_verts = person_data['smplx_verts']
                else:
                    frame_verts = person_data['verts']
                frame_cam = person_data['cam']
                crop_info = {'opt_cam_t': person_data['cam_t']}

                mesh_filename = None

                if args.save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                if args.empty_bg:
                    img, empty_img = renderer(
                            frame_verts,
                            img=[img, empty_img],
                            cam=frame_cam,
                            crop_info=crop_info,
                            color_type=color_type,
                            iwp_mode=False,
                            crop_img=False,
                            mesh_filename=mesh_filename
                        )
                else:
                    img_full = renderer(
                        frame_verts,
                        img=img_full,
                        cam=frame_cam,
                        crop_info=crop_info,
                        color_type=color_type,
                        iwp_mode=False,
                        crop_img=False,
                        mesh_type=args.render_model,
                        mesh_filename=mesh_filename
                    )

                    img_arm = renderer(
                        frame_verts[smpl2part],
                        faces=part_faces,
                        img=img_arm,
                        cam=frame_cam,
                        crop_info=crop_info,
                        color_type=color_type,
                        iwp_mode=False,
                        crop_img=False,
                        mesh_filename=mesh_filename
                    )

            if args.with_raw:
                img = np.concatenate([raw_img, img], axis=1)

            if args.empty_bg:
                img = np.concatenate([img, empty_img], axis=1)

            if args.vid_file is not None:
                imsave(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img_full)
                imsave(os.path.join(output_img_folder, 'arm', f'{frame_idx:06d}.png'), img_arm)
            else:
                imsave(os.path.join(output_img_folder, osp.split(img_fname)[-1][:-4]+'.png'), img_full)
                imsave(os.path.join(output_img_folder, 'arm', osp.split(img_fname)[-1][:-4]+'.png'), img_arm)                

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        if args.vid_file is not None:
            vid_name = osp.split(image_folder)[-1] if args.image_folder is not None else os.path.basename(video_file)
            save_name = f'{vid_name.replace(".mp4", "")}_result.mp4'
            save_name = os.path.join(output_path, save_name)

            print(f'Saving result video to {save_name}')
            images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
            images_to_video(img_folder=output_img_folder + '/arm', output_vid_file=save_name.replace(".mp4", "_arm.mp4"))
            images_to_video(img_folder=image_folder, output_vid_file=save_name.replace(".mp4", "_raw.mp4"))

            # remove temporary files
            shutil.rmtree(output_img_folder)
            shutil.rmtree(image_folder)

    print('================= END =================')

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)



    parser.add_argument('--img_file', type=str, default=None,
                        help='Path to a single input image')
    parser.add_argument('--vid_file', type=str, default=None,
                        help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='input image folder')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='output folder to write results')
    parser.add_argument('--tracking_method', type=str, default='pose', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--detector_checkpoint', type=str, default='shufflenetv2k30-wholebody',
                        help='detector checkpoint for openpifpaf')
    parser.add_argument('--detector_batch_size', type=int, default=1,
                        help='batch size of person detection')
    parser.add_argument('--detection_threshold', type=float, default=0.55,
                        help='pifpaf detection score threshold.')
    parser.add_argument('--single_person', action='store_true',
                        help='only one person in the scene.')
    parser.add_argument('--cfg_file', type=str, default='configs/pymafx_config.yaml',
                        help='config file path.')
    parser.add_argument('--pretrained_model', default=None,
                        help='Path to network checkpoint')
    parser.add_argument('--pretrained_body', default=None, help='Load a pretrained checkpoint for body at the beginning training') 
    parser.add_argument('--pretrained_hand', default=None, help='Load a pretrained checkpoint for hand at the beginning training') 
    parser.add_argument('--pretrained_face', default=None, help='Load a pretrained checkpoint for face at the beginning training') 

    parser.add_argument('--misc', default=None, type=str, nargs="*",
                        help='other parameters')
    parser.add_argument('--model_batch_size', type=int, default=8,
                        help='batch size for SMPL prediction')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')
    parser.add_argument('--render_vis_ratio', type=float, default=1.,
                        help='transparency ratio for rendered results')
    parser.add_argument('--render_part', type=str, default='arm',
                        help='render part mesh')
    parser.add_argument('--render_model', type=str, default='smplx', choices=['smpl', 'smplx'],
                        help='render model type')
    parser.add_argument('--with_raw', action='store_true',
                        help='attach raw image.')
    parser.add_argument('--empty_bg', action='store_true',
                        help='render meshes on empty background.')
    parser.add_argument('--use_gt', action='store_true',
                        help='use the ground truth tracking annotations.')
    parser.add_argument('--anno_file', type=str, default='',
                        help='path to tracking annotation file.')
    parser.add_argument('--render_ratio', type=float, default=1.,
                        help='ratio for render resolution')
    parser.add_argument('--recon_result_file', type=str, default='',
                        help='path to reconstruction result file.')
    parser.add_argument('--pre_load_imgs', action='store_true',
                        help='pred-load input images.')
    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()
    parse_args(args)

    print('Running demo...')
    run_demo(args)
