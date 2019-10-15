###############################################################################
# train_fns.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.15.2019
# Purpose: - This file provides training functions for both the GAN model
#            and the VAE model. The desired training funcion is chosen at
#            runtime. 
###############################################################################

# Imports
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

########################
#  Training Functions  #
########################
# GAN training function
def GAN_train_fn(G, D, G_D, z_fixed, loss_fn, config):
    '''
        GAN training function
        Does: Initializes the training function for the GAN model.
        Args: - G (Torch neural network): Generator model
              - D (Torch neural network): Discriminator model
              - G_D (Torch neural network, optional): Parallel model
              - z_fixed (Torch tensor): Fixed tensor of Gaussian noise for 
                                        sampling Generator function.
              - loss_fn (function): Torch loss function
    '''
    def train(x):
        '''
            GAN training function
            Does: Trains the GAN model
            Args: x (Torch tensor): Real data input image
            Returns: list of training metrics
        '''
        G.optim.zero_grad()
        D.optim.zero_grad()

        # Set up 'real' and 'fake' targets
        real_target = torch.ones(config['out_features']).to(config['device'])
        fake_target = torch.zeros(config['out_features']).to(config['device'])

        # Train D
        ## Move real batch onto GPU
        x = x.view(config['in_features']).to(config['device'])

        ## Generate fake data
        fake = torch.randn(config['batch_size'], config['z_dim'], device=config['device'])
        
        ## Detach so that gradients are not calc'd for G
        fake = G(fake).detach()

        ## 1.1 Train on real data
        real_pred  = D(x)
        real_error = loss_fn(real_pred, real_target) # Calculate error
        real_error.backward() # Backprop

        ## 1.2 Train on fake data
        fake_pred = D(fake)
        fake_error = loss_fn(fake_pred, fake_target) # Calculate error
        fake_error.backward() # Backprop

        ## Update D weights
        D.optim.step()

        # Train G
        ## Generate fake data
        fake = torch.randn(config['batch_size'], config['z_dim'], device=config['device'])

        ## Pass fake data through D
        G_pred = D((G(fake)))
        G_error = loss_fn(G_pred, real_target) # Calculate error
        G_error.backward() # Backprop

        # Update G weights
        G.optim.step()

        # Return training metric
        metrics = {'g_loss'     : float(G_error.item()),
                   'd_loss_real': float(real_error.item()),
                   'd_loss_fake': float(fake_error.item())}
        return metrics
    return train

# VAE training function
def VAE_train_fn():
    return 0
