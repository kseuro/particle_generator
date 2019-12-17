###############################################################################
# train.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.15.2019
# Purpose: - This file provides a model agnostic training routine. Training is
#            carried out as standard batch-to-batch training for both the
#            GAN and AE. Each model's respective training function is loaded
#            at runtime from train_fns.py
###############################################################################
# Imports
import os
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import time

# My stuff
import utils
import train_fns
import setup_model
from argparser import train_parser

def train(config):
    '''
        Model agnostic training function
        Does: - Takes a configuration dictionary and sets up training
                routine for GAN, AE, or EWM.
              - Writes model outputs, checkpoints, and metrics to disk during
                and after training.
        Args: config (dictionary): config dict of parsed command line args
        Returns: Nothing
    '''
    # Import the selected model
    model = __import__(config['model']) # GAN, AE, or EWM

    # Import appropriate training loop
    loop = train_fns.get_train_loop(config)

    # Instantiate the model and corresponding optimizer
    if (config['model'] == 'gan'):
        kwargs = setup_model.gan(model, config)
    elif (config['model'] == 'ae'):
        kwargs = setup_model.ae(model, config)
    elif (config['model'] == 'ewm'):
        emw_kwargs = utils.ewm_kwargs(config)
        G = model.Generator(emw_kwargs).to(config['gpu'])
        model_params = {'g_params': G.parameters()}
        G_optim = utils.get_optim(config, model_params)

    # Check if resuming from checkpoint
    if config['checkpoint']:
        '''
            This functionality should be written after state_dict saving
            conventions have been established for all three model training
            routines.
        '''
        pass

    # Set up progress bars for terminal output and enumeration
    dataloader = tqdm(utils.get_dataloader(config))
    epoch_bar  = tqdm([i for i in range(config['num_epochs'])])

    # Set up directories for saving training stats and outputs
    config = utils.directories(config)

    # Update key word arguments with final config dict and dataloader
    # Set up dicts for tracking progress
    history, best_stats = {}, {}
    times = {'epoch_times': [], 'tr_loop_times': []}
    kwargs.update({'config' : config,  'dataloader' : dataloader,
                   'history': history, 'best_stats' : best_stats,
                   'times'  : times})

    # If GAN or EWM set fixed random vector for sampling at the end of each epoch
    if (config['model'] != 'ae'):
        z_fixed = torch.randn(config['sample_size'], config['z_dim']).to(config['gpu'])
        kwargs.update( {'z_fixed' : z_fixed} )
    else:
        x_fixed = next(iter(dataloader))
        print(x_fixed.shape)
        kwargs.update( {'x_fixed' : x_fixed.view(config['batch_size'], 1, config['dataset'], config['dataset'])} )

    # Train model for specified number of epochs
    for epoch, _ in enumerate(epoch_bar):

        epoch_start = time.time()

        args = (epoch, epoch_start)

        history, best_stats, times = loop(*args, **kwargs)

    # Save training history and experiment config for evaluation and deploy
    utils.save_train_hist(history, best_stats, times, config)
    print("Training Complete")

def main():
    parser = train_parser()
    config = vars(parser.parse_args())
    train(config)

if __name__ == '__main__':
    main()
