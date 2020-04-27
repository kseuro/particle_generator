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
import setup_model
from argparser import train_parser

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
#     ewm.weights_init(G)
    G.train()
    
    # Setup model optimizer
    model_params = {'g_params': G.parameters()}
    G_optim = utils.get_optim(config, model_params)
    
    # Testing: MSE loss for conv_generator reconstruction
    loss_fn = nn.MSELoss().to(device)
    
    # Print G -- make sure it's right before continuing
    print(G)
    input('Press any key to continue')
    
    # Setup source of structured noise on GPU (Trained EWM_MLP_Generator Model)
    # Add these configs to the experiment source file
    if not config['ewm_root']:
        raise Exception('Path to trained EWM_MLP Model must be specified')

    # Print list of evaluated EWM model
    EWM_paths = []; EWM_root = config['ewm_root']
    for path in os.listdir(EWM_root):
        EWM_paths.append(os.path.join(EWM_root, path))
    print("-"*60)
    for i in range(len(EWM_paths)):
        EWM_name = EWM_paths[i].split('/')[-1]
        print("\n Exp_{}:".format(str(i)), EWM_name, '\n')
        print("-"*60)
    
    # Select the trained model
    model_num = input('Select EWM_MLP model (enter integer): ')
    EWM_dir = EWM_paths[int(model_num)]
    
    # Create the full path to the EWM model
    EWM_path = os.path.join(EWM_root, EWM_dir) + '/'
    print("Path to EWM Generator Model set as: \n{}".format(EWM_path))
    
    config_csv = EWM_path + "config.csv"
    config_df = pd.read_csv(config_csv, delimiter = ",")

    # Get the model architecture from config df
    n_layers = int(config_df[config_df['Unnamed: 0'].str.contains("n_layers")==True]['0'].values.item())
    n_hidden = int(config_df[config_df['Unnamed: 0'].str.contains("n_hidden")==True]['0'].values.item())
    l_dim    = int(config_df[config_df['Unnamed: 0'].str.contains("l_dim")==True]['0'].values.item())
    im_size  = int(config_df[config_df['Unnamed: 0'].str.contains("dataset")==True]['0'].values.item())
    z_dim    = int(config_df[config_df['Unnamed: 0'].str.contains("z_dim")==True]['0'].values.item())

    # Model kwargs
    ewm_kwargs = {'z_dim': z_dim, 'fc_sizes': [n_hidden]*n_layers, 'n_out': l_dim}

    # Send model to GPU
    Gz = ewm.ewm_G(**ewm_kwargs).to(device)
 
    # Load the model checkpoint
    # Get checkpoint name(s)
    EWM_checkpoint_path  = EWM_path + 'weights/'
    EWM_checkpoint_names = []
    for file in os.listdir(EWM_checkpoint_path):
        EWM_checkpoint_names.append(os.path.join(EWM_checkpoint_path, file))
    print("-"*60)
    for i in range(len(EWM_checkpoint_names)):
        name = EWM_checkpoint_names[i].split('/')[-1]
        print("\n {} :".format(str(i)), name, '\n')
        print("-"*60)
    file_num = input("Select a checkpoint file for EWM_MLP (enter integer): ")
    EWM_checkpoint = EWM_checkpoint_names[int(file_num)]

    # Load the model checkpoint
    # Keys: ['state_dict', 'epoch', 'optimizer']
    checkpoint = torch.load(EWM_checkpoint)

    # Load the model's state dictionary
    Gz.load_state_dict(checkpoint['state_dict'])

    # Use the code_vector model in evaluation mode -- no need for gradients here
    Gz.eval()
    print(Gz)
    input('Press any key to continue')

    # Set up full_dataloader (single batch)
    dataloader = utils.get_dataloader(config).to(device) # Full Dataloader
    dset_size  = len(dataloader)

    # Flatten the dataloader into a Tensor of shape [dset_size, l_dim]
#     dataloader = dataloader.view(dset_size, -1).to(device)

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

    # Compute how the input vectors need to be reshaped, based on conv_G input layer
    in_f  = G.main[0][2].in_channels; out_f = G.main[0][2].out_channels
    
    # Set the height and width of the feature maps. Note: Manually setting this to 8 is
    # hackish, but computing the actualy value requires knowing the number of layers in
    # the AutoEncoder whose code layer was used to train the generator model being loaded
    # here. I'm avoiding loading multiple paths and dataframes by simply setting it to 8, 
    # but maybe you can do better than I did...
    H = W = 8
    print('Vectors will be reshaped as: [{}] --> [{},{},{}]'.format(l_dim, in_f, H, W))
    
    # Set a fixed feature tensor for testing
    noise  = torch.randn(config['sample_size'], config['z_dim']).to(device)
    z_fixed = Gz(noise).view(-1, in_f, H, W).to(device)

    # Training Loop
    input('\nPress any key to launch -- good luck out there\n')

    # Set up progress bar for terminal output and enumeration
    epoch_bar  = tqdm([i for i in range(config['num_epochs'])])
    for epoch, _ in enumerate(epoch_bar):

        history['epoch'] = epoch

        # Set up memory lists: 
        #     - mu: simple feed-forward distribution 
        #     - transfer: transfer plan given by lists of indices
        # Rule-of-thumb: do not save the tensors themselves: instead, save the 
        #                data as a list and covert it to a tensor as needed.
        mu = [0] * config['mem_size']
        transfer = [0] * config['mem_size']
        mem_idx = 0

        # Compute the Optimal Transport Solver
        print("\nOptimal Transport Solver")
        ots_bar  = tqdm([i for i in range(dset_size//10)])
        for ots_iter, _ in enumerate(ots_bar):
            history['iter'] = ots_iter

            psi_optim.zero_grad()

            # Generate samples from cove_vector distribution
            z_batch = torch.randn(config['batch_size'], config['z_dim']).to(device)
            z_batch = Gz(z_batch).view(-1, in_f, H, W)

            # Push structured noise vector through convolutional generator
#             y_fake = G(z_batch).view(config['batch_size'], -1) # Flatten the output to match dataloader
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

            # Update memory tensors (lists)
            mu[mem_idx] = z_batch.data.cpu().numpy().tolist()
            transfer[mem_idx] = hit.data.cpu().numpy().tolist()
            mem_idx = (mem_idx + 1) % config['mem_size']

            # Update losses
            history['losses']['ot_loss'].append(loss.item())

            if (ots_iter % 50 == 0):
                avg_loss = np.mean(history['losses']['ot_loss'])
#                 print('OTS Iteration {} | Epoch {} | Avg Loss Value: {}'.format(ots_iter, epoch, round(avg_loss, 3)))
            if (ots_iter % 2000 == 0):
                # Occasionally save a random sample from the generator during OTS
                sample = y_fake[0:config['sample_size']].view(-1, 1, config['dataset'], config['dataset'])
                utils.save_sample(sample, epoch, ots_iter, config['random_samples'])
                
#                 # Display histogram stats
#                 hist_dict, stop = utils.update_histogram(transfer, history, config)
#                 # Emperical stopping criterion
#                 if stop:
#                     break

        # Compute the Optimal Fitting Transport Plan
        print("\nFitting Optimal Transport Plan")
        fit_bar  = tqdm([i for i in range(config['mem_size'])])
        for fit_iter, _ in enumerate(fit_bar):
            G_optim.zero_grad()
            
            # Retrieve stored batch of generated samples
#             z_batch = torch.tensor(mu[fit_iter]).view(-1, in_f, H, W).to(device)
            z_batch = torch.tensor(mu[fit_iter]).to(device)
            
            # Flatten the model output to match dataloader
#             y_fake = G(z_batch).view(config['batch_size'], -1)
            y_fake = G(z_batch)
            
            # Get Transfer plan from OTS: T(G_{t-1}(Gz))
            t_plan = torch.tensor(transfer[fit_iter])
            y0_hit = dataloader[t_plan].to(device)

            # Compute Wasserstein distance between G and T
#             G_loss = torch.mean(torch.abs(y0_hit - y_fake)) * l_dim
            G_loss = loss_fn(y_fake, y0_hit)
            
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

            if (fit_iter % 50 == 0):
                avg_loss = np.mean(history['losses']['fit_loss'])
#                 print('FIT Iteration {} | Epoch {} | Avg Loss Value: {}'.format(fit_iter, epoch, round(avg_loss,3)))
         
        # Save a fixed sample of the generator's output at the end of FIT
        sample = G(z_fixed).view(-1, 1, config['dataset'], config['dataset'])
        utils.save_sample(sample, epoch, fit_iter, config['fixed_samples'])

    # Save a checkpoint at end of training
    checkpoint = utils.get_checkpoint(history['epoch'], checkpoint_kwargs, config)
    utils.save_checkpoint(checkpoint, config)

    # Save training data to csv's after training end
    utils.save_train_hist(history, config, times=None, histogram=history['hist_dict'])
    print("Stop Counter Triggered {} Times".format(stop_counter))

    # For Spike
    print("See you, Space Cowboy")

def main():
    parser = train_parser()
    config = vars(parser.parse_args())
    train(config)

if __name__ == '__main__':
    main()