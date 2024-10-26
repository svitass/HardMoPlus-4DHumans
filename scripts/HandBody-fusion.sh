#!/bin/bash

file_to_delete1="../results/4dhuman_out.pkl"
file_to_delete2="../results/hamer_out.pkl"
checkpoint_human4d="../checkpoints/epoch=9-step=50000.ckpt"

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
source activate 4D_humans_copy_1.6
cd ../4D-Humans/
python demo_onlyone.py --img_folder ../results/example_vis --batch_size=48 --save_smpl --checkpoint "$checkpoint_human4d"
conda deactivate 
source activate hamer
cd ../hamer/
python demo_onlyone.py --img_folder ../results/example_vis  --batch_size=48 --save_smpl
conda deactivate 
source activate pymaf-x
cd ../PyMAF-X
python -m apps.render_arms --image_folder ../results/example_vis --detection_threshold 0.3 --pretrained_model data/pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt --misc TRAIN.BHF_MODE full_body MODEL.PyMAF.HAND_VIS_TH 0.1 --no_render --output_folder ../results
conda deactivate 
source activate 4D_humans_copy_1.6
cd ../4D-Humans
python mydemo1.py --checkpoint "$checkpoint_human4d"
