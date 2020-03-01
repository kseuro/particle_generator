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
from tqdm import tqdm, trange
from datetime import datetime

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
import ewm
import utils
import argparser
import setup_model

# torch.backends.cudnn.benchmark = True

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

    # Set up GPU device ordinal
    device = torch.device(config['gpu'])

    # Get model kwargs for convolutional generator
    config['model'] == 'ewm_conv'
    emw_kwargs = setup_model.ewm_kwargs(config)

    # Setup convolutional generator model on GPU
    G = ewm.ewm_convG(**emw_kwargs).to(device)

    # Setup source of structured noise on GPU
    # TODO: Compute the path to the ewm experiment that pertains to this
    #       experiment.
    # Add these configs to the experiment source file
    if not config['ewm_root']:
        config['ewm_root'] = config['save_root']
        if not config['ewm_target']:
            raise Exception('EWM target must be specified in order to compute path to experiment folder.')
        if 'conv' in config['ewm_target']:
            config['ewm_root'] += 'ewm_models/conv_ewm/'
        else
            config['ewm_root'] += 'ewm_models/mlp_ewm/'

    config_csv = EWM_path + "config.csv"
    config_df = pd.read_csv(config_csv, delimiter = ",")

    # Get the model architecture from config df
    n_layers = int(config_df[config_df['Unnamed: 0'].str.contains("n_layers")==True]['0'].values.item())
    n_hidden = int(config_df[config_df['Unnamed: 0'].str.contains("n_hidden")==True]['0'].values.item())
    l_dim    = int(config_df[config_df['Unnamed: 0'].str.contains("l_dim")==True]['0'].values.item())
    im_size  = int(config_df[config_df['Unnamed: 0'].str.contains("dataset")==True]['0'].values.item())
    z_dim    = int(config_df[config_df['Unnamed: 0'].str.contains("z_dim")==True]['0'].values.item())

    # Model kwargs
    # Height and Width of the bottleneck tensor depend on the number of times the
    # dim of the input data image has been reduced by half.
    H = W = config['dataset'] // n_layers
    ewm_kwargs = { 'z_dim': z_dim, 'fc_sizes': [n_hidden]*n_layers, 'n_out': l_dim*H*W }

    # Send model to GPU
    Gz = ewm.ewm_G(**ewm_kwargs).to(device)

    # Load the model checkpoint
    # Get checkpoint name(s)
    EWM_checkpoint_path  = config['ewm_root'] + 'weights/'
    EWM_checkpoint_names = []
    for file in os.listdir(EWM_checkpoint_path):
        EWM_checkpoint_names.append(os.path.join(EWM_checkpoint_path, file))
    print("-"*60)
    for i in range(len(EWM_checkpoint_names)):
        name = EWM_checkpoint_names[i].split('/')[-1]
        print("\n {} :".format(str(i)), name, '\n')
        print("-"*60)
    file_num = input("Select a checkpoint file (enter integer)")
    EWM_checkpoint = EWM_checkpoint_names[int(file_num)]

    # Load the model checkpoint
    # Keys: ['state_dict', 'epoch', 'optimizer']
    checkpoint = torch.load(EWM_checkpoint)

    # Load the model's state dictionary
    Gz.load_state_dict(checkpoint['state_dict'])

    # Use the code_vector model in evaluation mode -- no need for gradients here
    Gz.eval()
    print(G)
    input('Press any key to launch -- good luck out there')

    # Setup model optimizer
    model_params = {'g_params': G.parameters()}
    G_optim = utils.get_optim(config, model_params)

    # Set up full_dataloader (single batch)
    dataloader = utils.get_dataloader(config) # Full Dataloader
    dset_size  = len(dataloader)

    # Flatten the dataloader into a Tensor of shape [dset_size, l_dim]
    dataloader = dataloader.view(dset_size, -1).to(device)

    # Set up progress bar for terminal output and enumeration
    epoch_bar  = tqdm([i for i in range(config['num_epochs'])])

    # Set up psi optimizer
    psi = torch.zeros(dset_size, requires_grad=True).to(device).detach().requires_grad_(True)
    psi_optim = torch.optim.Adam([psi], lr=config['psi_lr'])

    # Set up directories for saving training stats and outputs
    config = utils.directories(config)

    # Set up dict for saving checkpoints
    checkpoint_kwargs = {'G':G, 'G_optim':G_optim}

    # Set up stats logging
    hist_dict = {'hist_min':[], 'hist_max':[], 'ot_loss':[]}
    losses    = {'ot_loss': [], 'fit_loss': []}
    history   = {'dset_size': dset_size, 'epoch': 0, 'iter': 0,
                 'losses'   : losses, 'hist_dict': hist_dict}
    config['early_end'] = (200, 320) # Empirical stopping criterion from EWM author

    stop_counter = 0

    # Training Loop
    for epoch, _ in enumerate(epoch_bar):

        history['epoch'] = epoch

        # Set up memory tensors: simple feed-forward distribution, transfer plan
        mu = torch.zeros(config['mem_size'], config['batch_size'], config['z_dim'])
        transfer = torch.zeros(config['mem_size'], config['batch_size'], dtype=torch.long)
        mem_idx = 0

        # Compute the Optimal Transport Solver
        for iter in range(1, dset_size//3):

            history['iter'] = iter

            psi_optim.zero_grad()

            # Generate samples from cove_vector distribution
            z_batch = torch.randn(config['batch_size'], config['z_dim']).to(device)
            z_batch = Gz(z_batch)

            # Push structured noise vector through convolutional generator
            y_fake = G(z_batch)

            # Compute the W1 distance between the model output and the target distribution
            score = my_ops.l1_t(y_fake, dataloader) - psi

            phi, hit = torch.max(score, 1)

            # Standard loss computation
            # This loss defines the sample mean of the marginal distribution
            # of the dataset. This is the only computation that generalizes.
            loss = -torch.mean(psi[hit])

            # Backprop
            loss.backward()
            psi_optim.step()

            # Update memory tensors
            mu[mem_idx] = z_batch
            transfer[mem_idx] = hit
            mem_idx = (mem_idx + 1) % config['mem_size']

            # Update losses
            history['losses']['ot_loss'].append(loss.item())

            if (iter % 500 == 0):
                avg_loss = np.mean(history['losses']['ot_loss'])
                print('OTS Iteration {} | Epoch {} | Avg Loss Value: {}'.format(iter,epoch,round(avg_loss, 3)))
            if (iter % 2000 == 0):
                # Display histogram stats
                hist_dict, stop = utils.update_histogram(transfer, history, config)
                # Emperical stopping criterion
                if stop:
                    break

            if epoch > 2: # min and max are swapped beacause loss is negative value
                if stop_max <= np.mean(history['losses']['ot_loss']) <= stop_min:
                    stop_counter += 1
                    break

        # Compute the Optimal Fitting Transport Plan
        for fit_iter in range(config['mem_size']):

            G_optim.zero_grad()

            # Retrieve stored batch of generated samples
            z_batch = mu[fit_iter].to(device)
            y_fake  = G(z_batch) # G'(z)

            # Get Transfer plan from OTS: T(G_{t-1}(z))
            y0_hit = dataloader[transfer[fit_iter]].to(device)

            # Compute Wasserstein distance between G and T
            G_loss = torch.mean(torch.abs(y0_hit - y_fake)) * config['l_dim']

            # Backprop
            G_loss.backward() # Gradient descent
            G_optim.step()

            # Update losses
            history['losses']['fit_loss'].append(G_loss.item())

            # Check if best loss value and save checkpoint
            if 'best_loss' not in history:
                history.update({ 'best_loss' : G_loss.item() })

            best = G_loss.item() < (history['best_loss'] * 0.5)
            if best:
                history['best_loss'] = G_loss.item()
                checkpoint = utils.get_checkpoint(history['epoch'], checkpoint_kwargs, config)
                utils.save_checkpoint(checkpoint, config)

            if (fit_iter % 500 == 0):
                avg_loss = np.mean(history['losses']['fit_loss'])
                print('FIT Iteration {} | Epoch {} | Avg Loss Value: {}'.format(fit_iter, epoch, round(avg_loss,3)))

    # Save a checkpoint at end of training
    checkpoint = utils.get_checkpoint(history['epoch'], checkpoint_kwargs, config)
    utils.save_checkpoint(checkpoint, config)

    # Save training data to csv's after training end
    utils.save_train_hist(history, config, times=None, histogram=history['hist_dict'])
    print("Stop Counter Triggered {} Times".format(stop_counter))

    # For Spike
    print("See you, Space Cowboy")
