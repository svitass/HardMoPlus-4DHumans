# baseline evaluation on freeman
python eval/eval_hmr.py --dataset 'FREEMAN-SIDEVIEW-TEST' --batch_size 64 --checkpoint data/baseline/pro_hmr/data/checkpoint.pt --results_file eval_sideview_results/eval_prohmr_freeman.csv  --device_name="cuda:3" --model_type pro_hmr

python eval/eval_hmr.py --dataset 'FREEMAN-SIDEVIEW-TEST' --batch_size 64 --checkpoint data/baseline/hmr_hardmo/checkpoints/epoch=24-step=300000.ckpt --results_file eval_sideview_results/eval_hmr_hardmo_freeman.csv --device_name="cuda:3" --model_type hmr

python eval/eval_hmr.py --dataset 'FREEMAN-SIDEVIEW-TEST' --batch_size 64 --checkpoint data/baseline/hmr2b/checkpoints/epoch=35-step=1000000.ckpt --results_file eval_sideview_results/eval_hmr2b_freeman.csv --device_name="cuda:3" --model_type hmr2

python eval/eval_hmr.py --dataset 'FREEMAN-SIDEVIEW-TEST' --batch_size 64 --checkpoint data/baseline/hmr2a/checkpoints/epoch=10-step=25000.ckpt  --results_file eval_sideview_results/eval_hmr2a_freeman.csv --device_name="cuda:3" --model_type hmr2

python eval/eval_hmr.py --dataset 'FREEMAN-SIDEVIEW-TEST' --batch_size 64 --checkpoint data/baseline/hmr2_hardmo/checkpoints/epoch=25-step=85000.ckpt  --results_file eval_sideview_results/eval_hmr2_hardmo_freeman.csv --device_name="cuda:3" --model_type hmr2

python eval/eval_hmr.py --dataset 'FREEMAN-SIDEVIEW-TEST' --batch_size 64 --checkpoint data/baseline/hmr2_hardmoplus/checkpoints/epoch=25-step=140000.ckpt  --results_file eval_sideview_results/eval_hmr2_hardmoplus_freeman.csv --device_name="cuda:3" --model_type hmr2
