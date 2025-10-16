#!/bin/bash

# RMSCM多模态手势识别训练启动脚本 - SwanLab版本

# 受试者10训练 - 使用SwanLab监控
python3 train_swanlab.py \
    --data_dir processed_data \
    --subject 10 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --optimizer adam \
    --scheduler step \
    --step_size 30 \
    --gamma_lr 0.1 \
    --feature_dim 64 \
    --hidden_dim 64 \
    --dropout 0.5 \
    --alpha 1.0 \
    --beta 1.0 \
    --gamma 1.0 \
    --num_workers 4 \
    --early_stopping \
    --patience 15 \
    --grad_clip 1.0 \
    --eval_freq 1 \
    --device cuda:0 \
    --checkpoint_dir checkpoints/S10_swanlab \
    --log_dir logs/S10_swanlab \
    --print_freq 10 \
    --save_freq 10 \
    --use_tensorboard \
    --use_swanlab \
    --swanlab_project "Momo-Gesture-Recognition"

