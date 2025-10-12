python run_seed_search.py \
  --gpus 0,1,2,3 \
  --max-jobs-per-gpu 1 \
  --preset configs/csc_clam_cl.yaml \
  --cl-method prev \
  --buffer-size 42 \
  --exp-name-prefix mix_wn_logs \
  --log-dir mix_wn_logs \
  --seeds 1,2,3,4 \



  python run_seed_search.py \
  --gpus 4,5,6,7 \
  --max-jobs-per-gpu 1 \
  --preset configs/csc_clam_cl.yaml \
  --cl-method prev \
  --buffer-size 42 \
  --exp-name-prefix mix_balance_logs \
  --log-dir mix_balance_logs \
  --seeds 1,2,3,4 \


