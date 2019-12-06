#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--gpu 1 \
--dataset 512 \
--batch_size 24 \
--num_epochs 100 \
--sample_size 8 \
--shuffle True \
--drop_last True \
--num_workers 8 \
--model ae \
--n_layers 4 \
--l_dim 10 \
--ae_lr 1e-4 \
--ae_opt adam \
--beta 0.5 \
--data_root /media/disk1/kai/larcv_png_data/ \
--save_root /media/hdd1/kai/particle_generator/experiments/
