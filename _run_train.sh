#!/bin/bash
# /root/CFANet-main/_run_train.sh
# Fresh start: batch=8, LR=5e-5 (the working config)
set -e
cd /root/CFANet-main
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec python3 -u train.py \
    --epoch 20 \
    --batchsize 8 \
    --trainsize 256 \
    --crop_size 384 \
    --decay_epoch 12 \
    --save_epoch 5 \
    --log_every 50 \
    --train_root ./data/traindata \
    --val_root   ./data/traindata \
    --test_root  ./data/testdata \
    --mask_combine or \
    --train_ratio 0.8 --val_ratio 0.2 --test_ratio 0.0 \
    --resplit_seed 42 \
    --w_ft 1.5 --w_bce 0.3 --w_dice 0.5 \
    --ft_gamma 0.75 --boundary_k 7 \
    --size_rates 1 \
    --grad_clip 1.0 --warmup_steps 200 \
    --deep_sup_w 1,1,1,1 \
    --lr 5e-5
