#!/bin/bash

file_to_delete1="../results/4dhuman_out.pkl"
file_to_delete2="../results/hamer_out.pkl"
checkpoint_human4d="/data/boning_zhang/4D-Humans/logs/freeman_score_hamer/runs/finetune_on_hardmoplus/checkpoints/epoch=9-step=50000.ckpt"

if [ -f "$file_to_delete1" ]; then
    rm "$file_to_delete1"
    echo "$file_to_delete1 has been deleted."
else
    echo "$file_to_delete1 does not exist."
fi

if [ -f "$file_to_delete2" ]; then
    rm "$file_to_delete2"
    echo "$file_to_delete2 has been deleted."
else
    echo "$file_to_delete2 does not exist."
fi
# Activate the first conda environment
source activate 4D_humans_copy_1.6
# Run the first Python script
cd /data/boning_zhang/4D-Humans/
python demo_onlyone.py --img_folder /data/boning_zhang/blend_body_hand_pipline/example_vis --batch_size=48 --save_smpl --checkpoint "$checkpoint_human4d"

conda deactivate 

source activate hamer

cd /data/boning_zhang/hamer/
python demo_onlyone.py --img_folder /data/boning_zhang/blend_body_hand_pipline/example_vis  --batch_size=48 --save_smpl

#python hand_body_demo.py --img_folder /data/boning_zhang/blend_body_hand_pipline/example_data --out_folder /data/boning_zhang/blend_body_hand_pipline/demo_out --batch_size=48  --save_render --full_frame

conda deactivate 
# Activate the second conda environment
source activate pymaf-x
cd /data/boning_zhang/PyMAF-X
python -m apps.render_arms --image_folder /data/boning_zhang/blend_body_hand_pipline/example_vis --detection_threshold 0.3 --pretrained_model data/pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt --misc TRAIN.BHF_MODE full_body MODEL.PyMAF.HAND_VIS_TH 0.1 --no_render --output_folder /data/boning_zhang/blend_body_hand_pipline

conda deactivate 
# Activate the second conda environment
source activate 4D_humans_copy_1.6

cd /data/boning_zhang/4D-Humans
CUDA_VISIBLE_DEVICES=7 python mydemo1.py --checkpoint "$checkpoint_human4d"

#python demo.py --img_folder /data/boning_zhang/blend_body_hand_pipline/example_data --out_folder /data/boning_zhang/blend_body_hand_pipline/4dhuman_demo_out --batch_size=48 --side_view --save_mesh --full_frame
