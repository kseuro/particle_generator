#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
--gpu 1 \
--dataset 64 \
--batch_size 128 \
--num_epochs 7500 \
--sample_size 16 \
--shuffle True \
--drop_last True \
--num_workers 8 \
--model ae \
--n_layers 4 \
--l_dim 256 \
--ae_lr 1e-4 \
--ae_opt adam \
--beta 0.5 \
--data_root /media/hdd1/kai/particle_generator/larcv_data/train/ \
--save_root /media/hdd1/kai/particle_generator/experiments/
