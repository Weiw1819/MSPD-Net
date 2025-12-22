#!/bin/bash

port=29171
crop_size=512

file=scripts/dist_train_coco.py
config=configs/coco_attn_reg.yaml

CUDA_VISIBLE_DEVICES=2 echo python -m torch.distributed.launch --nproc_per_node=1 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_attn_coco
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_attn_coco
