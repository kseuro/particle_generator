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

    # If training Generator model using EWM algo, switch to
    # EWM training routine defined in train_ewm.py
    if config['model'] == 'ewm':
        from train_ewm import train as EWM_train
        EWM_train(config)
        return

    # Import appropriate training loop
    loop = train_fns.get_train_loop(config)

    # Instantiate the model and corresponding optimizer
    if config['model'] == 'gan':
        kwargs = setup_model.gan(model, config)
    else:
        kwargs = setup_model.ae(model, config)

    # Set up progress bars for terminal output and enumeration
    dataloader = tqdm(utils.get_dataloader(config))
    epoch_bar  = tqdm([i for i in range(config['num_epochs'])])

    # Set up directories for saving training stats and outputs
    config = utils.directories(config)

    # Update key word arguments with final config dict and dataloader
    # Set up dicts for tracking progress
    history, best_stat = {}, {}
    times = {'epoch_times': [], 'tr_loop_times': []}
    kwargs.update({'config' : config,  'dataloader' : dataloader,
                   'history': history, 'best_stat'  : best_stat,
                   'times'  : times})

    # If GAN, set fixed random vector for sampling at the end of each epoch
    if (config['model'] == 'gan'):
        z_fixed = torch.randn(config['sample_size'], config['z_dim']).to(config['gpu'])
        kwargs.update( {'z_fixed' : z_fixed} )

    # The variable 'best' will keep track of the previous best_stat and
    # be updated only once the best_stat changes to a new, lower value.
    best = None

    # Train model for specified number of epochs
    for epoch, _ in enumerate(epoch_bar):

        epoch_start = time.time()

        args = (epoch, epoch_start)

        history, best_stat, times = loop(*args, **kwargs)

        # Check losses starting after 5000 epochs and determine if
        # the current model is the best model state
        if (epoch > 5000) and (epoch % 250 == 0):
            for key in best_stat:
                if best is None:
                    best = best_stat[key]
                    checkpoint = utils.get_checkpoint(epoch, kwargs, config)
                    utils.save_checkpoint(checkpoint, config)
                if round(best_stat[key], 5) < round(best, 5):
                    best = best_stat[key]
                    checkpoint = utils.get_checkpoint(epoch, kwargs, config)
                    utils.save_checkpoint(checkpoint, config)

    # Save training history and experiment config for evaluation and deploy
    utils.save_train_hist(history, config, times=times, histogram=None)

    # Save one last checkpoint
    checkpoint = utils.get_checkpoint(epoch, kwargs, config)
    utils.save_checkpoint(checkpoint, config)

    # The Force will be with you - always
    print("Your training is complete.")
    print("But you are not a Jedi yet.")

def main():
    parser = train_parser()
    config = vars(parser.parse_args())
    train(config)

if __name__ == '__main__':
    main()
