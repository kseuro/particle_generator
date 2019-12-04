#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--gpu 0 \
--MNIST True \
--batch_size 128 \
--num_epochs 100 \
--sample_size 16 \
--shuffle True \
--drop_last True \
--num_workers 8 \
--model ae \
--n_layers 2 \
--l_dim 9 \
--ae_lr 1e-4 \
--ae_opt adam \
--beta 0.5 \
--save_root /media/hdd1/kai/particle_generator/experiments/
