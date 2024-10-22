import argparse
import os
#os.environ['PYOPENGL_PLATFORM'] == 'egl'
from pathlib import Path
import traceback
from typing import Optional
from yacs.config import CfgNode as CN

import pandas as pd
import torch
from filelock import FileLock
import sys
sys.path.append('/data/boning_zhang/HardMoPlus-4DHumans')
from hmr2.configs import dataset_eval_config
from hmr2.datasets import create_dataset
from hmr2.utils import Evaluator, recursive_to
from tqdm import tqdm
##print("test")
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT, HMR, Pro_HMR
import hydra
from hmr2.configs import get_config
from collections import OrderedDict

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--results_file', type=str, default='/data/boning_zhang/4D-Humans/eval_results/eval_freeman_p2.csv', help='Path to results file.')
    parser.add_argument('--dataset', type=str, default='H36M-VAL-P2,3DPW-TEST,LSP-EXTENDED,POSETRACK-VAL,COCO-VAL', help='Dataset to evaluate') # choices=['H36M-VAL-P2', '3DPW-TEST', 'MPI-INF-TEST']
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--device_name', default='cuda:4',help='name of device, cuda or cpu')
    parser.add_argument('--model_type', type=str, default='hmr', help='model_type')

    args = parser.parse_args()

    # make save dir
    save_dir = os.path.dirname(args.results_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Download and load checkpoints
    if args.model_type == 'hmr2':
        model, model_cfg = load_hmr2(args.checkpoint)
    elif args.model_type == 'hmr':
        model_cfg = "data/baseline/hmr_hardmo/model_config.yaml"
        model_cfg = get_config(model_cfg, update_cachedir=True)
        model = HMR(model_cfg).load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg,  map_location='cpu')
    elif args.model_type == 'pro_hmr':
        model_cfg = "data/baseline/pro_hmr/pro_hmr.yaml"
        model_cfg = get_config(model_cfg, update_cachedir=True)
        model = Pro_HMR(model_cfg).load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg,  map_location='cpu')
    
    # Setup HMR2.0 model
    device = torch.device(args.device_name) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load config and run eval, one dataset at a time
    print('Evaluating on datasets: {}'.format(args.dataset), flush=True)
    for dataset in args.dataset.split(','):
        dataset_cfg = dataset_eval_config()[dataset]
        args.dataset = dataset
        run_eval(model, model_cfg, dataset_cfg, device, args)

def run_eval(model, model_cfg, dataset_cfg, device, args):
    # Create dataset and data loader
    dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    # List of metrics to log
    if args.dataset in ['H36M-VAL-P2']:
        metrics = ['mode_re', 'mode_mpjpe']
        pck_thresholds = None
    
    if args.dataset in ['HARDAK-HAND-TRAIN-correct-pose','3DPW-TEST']:
        metrics = ['mode_kpl2', 'mode_re', 'mode_mpjpe']
        pck_thresholds = [0.01,0.05]
        
    if args.dataset in ['LSP-EXTENDED', 'POSETRACK-VAL', 'COCO-VAL', 'AIST-TEST', 'HARDAK-BIG-TEST', "HARDAK-BIG-TEST-OPTIM",'3DPW-TEST-CLIFF', 'Hardmo-foot-p2-test', 'Hardmo-foot-p2-eft']:
        metrics = ['mode_kpl2', 'mode_re', 'mode_mpjpe']
        pck_thresholds = [0.01,0.05, 0.1]
    
    if args.dataset in ['HARDAK', 'HARDAK-BIG','HARDAK-hand', 'HARDAK-FOOT-P2', 'HARDAK-FOOT-P1', '3DPW-TEST-FOOT', 'Bedlam-foot-test', 'FREEMAN-TEST', 'FREEMAN-SIDEVIEW-TEST']:
        metrics = ['mode_re', 'mode_mpjpe', 'mode_kpl2', 'mode_foot_mpjpe', 'mode_foot_re', 'mode_angle_bias']
        pck_thresholds = [0.01, 0.05]
    if args.dataset in ['FREEMAN_SCORE_HAMER-TEST']:
        metrics = ['mode_angle_bias','mode_rotmat_loss']
        pck_thresholds = None
    # Setup evaluator object
    evaluator = Evaluator(
        dataset_length=int(1e8), 
        keypoint_list=dataset_cfg.KEYPOINT_LIST, 
        pelvis_ind=model_cfg.EXTRA.PELVIS_IND, 
        metrics=metrics,
        pck_thresholds=pck_thresholds,
    )
    

    # Go over the images in the dataset.这个地方那我们直接令batch_size等于
    try:
        for i, batch in enumerate(tqdm(dataloader)):
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            evaluator(out, batch)
            if i % args.log_freq == args.log_freq - 1:
                evaluator.log()
        evaluator.log()
        error = None
    except (Exception, KeyboardInterrupt) as e:
        traceback.print_exc()
        error = repr(e)
        i = 0

    # Append results to file
    metrics_dict = evaluator.get_metrics_dict()
    save_eval_result(args.results_file, metrics_dict, args.checkpoint, args.dataset, error=error, iters_done=i, exp_name=args.exp_name)


def save_eval_result(
    csv_path: str,
    metric_dict: float,
    checkpoint_path: str,
    dataset_name: str,
    # start_time: pd.Timestamp,
    error: Optional[str] = None,
    iters_done=None,
    exp_name=None,
) -> None:
    """Save evaluation results for a single scene file to a common CSV file."""

    timestamp = pd.Timestamp.now()
    exists: bool = os.path.exists(csv_path)
    exp_name = exp_name or Path(checkpoint_path).parent.parent.name

    # save each metric as different row to the csv path
    metric_names = list(metric_dict.keys())
    metric_values = list(metric_dict.values())
    N = len(metric_names)
    df = pd.DataFrame(
        dict(
            timestamp=[timestamp] * N,
            checkpoint_path=[checkpoint_path] * N,
            exp_name=[exp_name] * N,
            dataset=[dataset_name] * N,
            metric_name=metric_names,
            metric_value=metric_values,
            error=[error] * N,
            iters_done=[iters_done] * N,
        ),
        index=list(range(N)),
    )

    # Lock the file to prevent multiple processes from writing to it at the same time.
    lock = FileLock(f"{csv_path}.lock", timeout=10)
    with lock:
        df.to_csv(csv_path, mode="a", header=not exists, index=False)

if __name__ == '__main__':
    main()
