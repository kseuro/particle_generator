#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1,2 python train.py \
--gpu 1 \
--dataset 128 \
--batch_size 256 \
--num_epochs 1000 \
--sample_size 16 \
--shuffle True \
--drop_last True \
--num_workers 8 \
--model ae \
--n_layers 4 \
--l_dim 256 \
--ae_lr 1e-4 \
--ae_opt adam \
--loss_fn bce \
--beta 0.5 \
--data_root /media/hdd1/kai/particle_generator/larcv_data/train/ \
--save_root /media/hdd1/kai/particle_generator/experiments/
