#!/bin/bash

file_to_delete1="./4dhuman_out.pkl"
file_to_delete2="./hamer_out.pkl"
checkpoint_human4d="./logs/freeman_score_hamer/runs/finetune_on_hardmoplus/checkpoints/epoch=9-step=50000.ckpt"
img_folder="../HandBody-fusion/example_vis"
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
source activate hamer
cd ../hamer/
python demo_onlyone.py --img_folder "$img_folder"  --batch_size=48 --save_smpl --save_mano_path "../HandBody-fusion/hamer_out.pkl"
conda deactivate

source activate human4d
cd ../4D-Humans/
python demo_onlyone.py --img_folder "$img_folder" --batch_size=48 --save_smpl --checkpoint "$checkpoint_human4d" --save_smpl_path "../HandBody-fusion/4dhuman_out.pkl"
conda deactivate 

source activate pymaf-x
cd ../PyMAF-X
python -m apps.render_arms --image_folder "$img_folder" --detection_threshold 0.3 --pretrained_model data/pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt --misc TRAIN.BHF_MODE full_body MODEL.PyMAF.HAND_VIS_TH 0.1 --no_render --output_folder ../HandBody-fusion
conda deactivate

source activate human4d
cd ../4D-Humans
python mydemo.py --checkpoint "$checkpoint_human4d"
