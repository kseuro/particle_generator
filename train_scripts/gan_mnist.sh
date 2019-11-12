#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
# Data loading
--gpu 0 \
--MNSIT True \
--batch_size 32 \
--num_epochs 1 \
--sample_size 8 \
--shuffle True \
--drop_last True \
--num_workers 8 \
# Model Settings
--model gan \
--n_hidden 512 \
--n_layers 4 \
--g_lr 1e-4 \
--g_opt adam \
--z_dim 100 \
--d_lr 1e-4 \
--d_opt adam \
# Shared model settings
--beta 0.5 \
# Directories
--save_root /media/hdd1/kai/particle_generator/experiments/