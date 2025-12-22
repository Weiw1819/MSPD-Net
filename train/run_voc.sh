#!/bin/bash


port=27978
crop_size=320

file=scripts/dist_train_voc.py
config=configs/voc_attn_reg.yaml



CUDA_VISIBLE_DEVICES=1,2,3 echo python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_voc
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config
--pooling gmp --crop_size $crop_size --work_dir work_dir_voc



