############################################################################
# train.py
# Author: Kai Kharpertian
# Tufts University - Department of Physics
# 01.08.2020
# - This script instantiates G as a multi-layer perceptron and computes
#       an optimal transport plan to fit G to the LArCV data distribution.
# - Use of a full dataloader is due to the fact that methods that form only
#       batch-to-batch transports using samples from a feed-forward
#       distribution are biased and do not exactly minimize the
#       Wasserstein distance.
############################################################################
# Implementation of Explicit Wasserstein Minimization described in:
# @article{1906.03471,
#   Author = {Yucheng Chen and Matus Telgarsky and Chao Zhang and Bolton Bailey
#             and Daniel Hsu and Jian Peng},
#   Title  = {A gradual, semi-discrete approach to generative network training
#             via explicit Wasserstein minimization},
#   Year   = {2019},
#   Eprint = {arXiv:1906.03471},
# }
############################################################################
# Algorithm 1: Optimal Transport Solver
# Input: Feed-forward distribution from G, training dataset
# Output: psi (optimal transport solver)
# This algorithm operates over the whole dataset and is O(N) complex
############################################################################
# Algorithm 2: Fitting Optimal Transport Plan
# Input: Sampling distribution, old generator function, Transfer plan
# Output: new generator function with updated parameters
############################################################################

# Sys Imports
import os
import time
import errno
import shutil
from   datetime import datetime

# Python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from   statistics import mean

# Torch
import torch
import torch.nn as nn
import torchvision.utils
from   torch.nn         import init
from   torch.utils.data import DataLoader
from   torchvision      import transforms
from   torchvision      import datasets as dset

# My stuff
import utils
import ewm
import argparser
import setup_model

torch.backends.cudnn.benchmark = True

def train(config):
    '''
        Training function for EWM generator model.
    '''
    # Create python version of cpp operation
    # (Credit: Chen, arXiv:1906.03471, GitHub: https://github.com/chen0706/EWM)
    from torch.utils.cpp_extension import load
    my_ops = load(name = "my_ops",
                  sources = ["W1_extension/my_ops.cpp",
                             "W1_extension/my_ops_kernel.cu"],
                  verbose = False)
    import my_ops

    # Update the config data_root to point to desired set of code vectors
    config['data_root'] += "code_vectors_{}_{}/".format(config['dataset'], config['l_dim'])

    # Get model kwargs
    emw_kwargs = setup_model.ewm_kwargs(config)

    # Setup model on GPU
    G = ewm.Generator(**emw_kwargs).to(config['gpu'])
    G.weights_init()

    # Setup model optimizer
    model_params = {'g_params': G.parameters()}
    G_optim = utils.get_optim(config, model_params)

    # Set up progress bars for terminal output and enumeration
    dataloader = utils.get_dataloader(config) # Full Dataloader
    dataloader = dataloader.view(len(dataloader), -1).to(gpu) # Flatten (may not be necessary)
    n_dim      = len(dataloader)
    epoch_bar  = tqdm([i for i in range(config['num_epochs'])])

    # Set up psi optimizer
    psi = torch.zeros(n_dim, requires_grad=True).to(config['gpu'])
    psi_optim = torch.optim.Adam([psi], lr=config['psi_lr'])

    # Set up directories for saving training stats and outputs
    config = utils.directories(config)

    # Set up dict for saving checkpoints
    checkpoint_kwargs = {'G':G, 'G_optim':G_optim}

    # Set up stats logging
    hist_dict = {'hist_min':[], 'hist_max':[], 'ot_loss':[]}
    losses    = {'ot_loss': [], 'fit_loss': []}
    history   = {'n_dim': n_dim, 'epoch': 0, 'iter': 0,
                 'losses':losses, 'hist_dict': hist_dict}
    config['early_end'] = (200, 320) # Expirical stopping criterion

    # Training Loop
    for epoch, _ in enumerate(epoch_bar):

        history['epoch'] = epoch

        # Set up memory tensors: simple feed-forward distribution, transfer plan
        mu = torch.zeros(config['mem_size'], config['batch_size'], config['z_dim'])
        transfer = torch.zeros(config['mem_size'], config['batch_size'], dtype=torch.long)
        mem_idx = 0

        # Compute the Optimal Transport Solver over every training example
        for iter in range(n_dim):

            history['iter'] = iter

            psi_optim.zero_grad()

            # Generate samples from feed-forward distribution
            z_batch = torch.randn(config['batch_size'], config['z_dim']).to(config['gpu'])
            y_fake  = G(z_batch) # [B, n_dim]

            # Compute the W1 distance between the model output and the target distribution
            score = my_ops.l1_t(y_fake, dataloader) - psi

            phi, hit = torch.max(score, 1)

            loss = -torch.mean(psi[hit]) # equiv. to loss

            # Backprop
            loss.backward() # Gradient ascent
            opt_psi.step()

            # Update memory tensors
            mu[mem_idx] = z_batch
            transfer[mem_idx] = hit
            mem_idx = (mem_idx + 1) % config['mem_size']

            # Update losses
            history['losses']['ot_loss'].append(loss.item())

            if (iter % 500 == 0):
                print('OTS Iteration {} | Epoch {}'.format(iter, epoch))
            if (iter % 2000 == 0):
                # Display histogram stats
                hist_dict, stop = utils.update_histogram(transfer, history, config)
                # Emperical stopping criterion
                if stop:
                    break

        # Compute the Optimal Fitting Transport Plan
        for fit_iter in range(config['mem_size']):

            G_optim.zero_grad()

            # Retrieve stored batch of generated samples
            z_batch = mu[fit_iter].to(config['gpu'])
            y_fake  = G(z_batch) # G'(z)

            # Get Transfer plan from OTS: T(G_{t-1}(z))
            y0_hit = dataloader[transfer[fit_iter]].to(config['gpu'])

            # Compute Wasserstein distance between G and T
            G_loss = torch.mean(torch.abs(y0_hit - y_fake)) * config['l_dim']

            # Backprop
            G_loss.backward() # Gradient descent
            G_optim.step()

            # Update losses
            history['losses']['fit_loss'].append(G_loss.item())

            # Check if best loss value and save checkpoint
            if not history['best_loss']:
                history.update({ 'best_loss' : G_loss.item() })

            best = G_loss.item() < (history['best_loss'] * 0.5)
            if best:
                history['best_loss'] = G_loss.item()
                checkpoint = utils.get_checkpoint(history['epoch'], checkpoint_kwargs, config)
                utils.save_checkpoint(checkpoint, config)

    # Save a checkpoint at end of training
    checkpoint = utils.get_checkpoint(history['epoch'], checkpoint_kwargs, config)
    utils.save_checkpoint(checkpoint, config)

    # Save training data to csv's after training end
    utils.save_train_hist(history, config, times=None, histogram=history['hist_dict'])

    # For Aiur
    print("Your thoughts betray you.")
    print("I see you have an appetite for destruction.")
    print("And you have learned to use your illusion.")
    print("But I find your lack of control disturbing.")