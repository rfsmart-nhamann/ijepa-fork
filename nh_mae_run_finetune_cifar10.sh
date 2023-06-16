#!/bin/sh

DATA_PATH='data/cifar10'
DATA_SET='CIFAR10'
PRETRAIN_CHKPT='pretrained/mae_pretrain_vit_base.pth'

python main_finetune.py \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune $PRETRAIN_CHKPT \
    --epochs 50 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 \
    --dist_eval --data_path $DATA_PATH --data_set $DATA_SET
