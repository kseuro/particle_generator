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
    # May leave option open to model parallelization later (G_D)
    if (config['model'] == 'gan'):
        # Get G and D kwargs based on command line inputs
        g_kwargs, d_kwargs = utils.gan_kwargs(config)

        # Set up models on GPU
        G = model.Generator(**g_kwargs).to(config['gpu'])
        D = model.Discriminator(**d_kwargs).to(config['gpu'])

        # Initialize model weights
        G.weights_init()
        D.weights_init()

        # Set up model optimizer functions
        model_params = {'g_params': G.parameters(),
                        'd_params': D.parameters()}
        G_optim, D_optim = utils.get_optim(config, model_params)

        # Set up loss function
        loss_fn = nn.BCELoss().to(config['gpu'])

        # Set up training function
        train_fn = train_fns.GAN_train_fn(G, D, G_optim, D_optim, loss_fn,
                                           config, G_D=None)

        # Set up dicts for tracking progress
        history, best_stats = {}, {}
        times = {'epoch_times' : [], 'tr_loop_times' : []}
        kwargs = {'G':G, 'G_optim':G_optim, 'D':D, 'D_optim':D_optim,
                  'train_fn':train_fn, 'history': history, 'best_stats':best_stats,
                  'times':times}
    elif (config['model'] == 'ae'):
        # Get model kwargs
        ae_kwargs, config = utils.ae_kwargs(config)

        # Set up model on GPU
        AE = model.AutoEncoder(**ae_kwargs).to(config['gpu'])

        print(AE)
        input('Press any key to launch')

        # Initialize the weights
        # AE.weights_init()

        # Set up model optimizer function
        model_params = {'ae_params' : AE.parameters()}
        AE_optim = utils.get_optim(config, model_params)

        # Set up loss function
        loss_fn = nn.MSELoss().to(config['gpu'])

        # Set up training function
        train_fn = train_fns.AE_train_fn(AE, AE_optim, loss_fn, config)

        # Set up dicts for tracking progress
        history, best_stats = {}, {}
        times = {'epoch_times': [], 'tr_loop_times': []}

        kwargs = {'AE'         : AE,         'AE_optim': AE_optim,
                  'train_fn'   : train_fn,   'history' : history,
                  'best_stats' : best_stats, 'times'   : times}
    elif (config['model'] == 'ewm'):
        emw_kwargs = utils.ewm_kwargs(config)
        G = model.Generator(emw_kwargs).to(config['gpu'])
        model_params = {'g_params': G.parameters()}
        G_optim = utils.get_optim(config, model_params)

        # Set up dicts for tracking progress

    # Check if resuming from checkpoint
    if config['checkpoint']:
        '''
            This functionality should be written after state_dict saving
            conventions have been established for all three model training
            routines.
        '''
        pass

    # Create dataloader object - either batch-to-batch, full, or MNSIT
    # TODO: This may need to be modified after AE training is complete
    #       in order to add ability to load model latent space as target
    #       dataset for EWM algorithm
    dataloader = utils.get_dataloader(config)

    # Set up progress bars for terminal output and enumeration
    dataloader = tqdm(dataloader)
    epoch_bar = tqdm([i for i in range(config['num_epochs'])])

    # Set up directories for saving training stats and outputs
    config = utils.directories(config)

    # Update key word arguments with final config dict and dataloader
    kwargs.update({'config' : config, 'dataloader' : dataloader})

    # Set fixed random vector for sampling at the end of each epoch
    # if training GAN model or EWM model
    if (config['model'] != 'ae'):
        z_fixed = torch.randn(config['sample_size'], config['z_dim']).to(config['gpu'])
        kwargs.update( {'z_fixed' : z_fixed} )

    # Train model for specified number of epochs
    for epoch, _ in enumerate(epoch_bar):
        epoch_start = time.time()
        args = (epoch, epoch_start)
        if (config['model'] == 'gan'):
            history, best_stats, times = loop(*args, **kwargs)
        elif (config['model'] == 'ae'):
            history, best_stats, times = loop(*args, **kwargs)
        elif (config['model'] == 'ewm'):
            pass
        else:
            raise Exception("Error in training loop!")

    # Save training history and experiment config for evaluation and deploy
    utils.save_train_hist(history, best_stats, times, config)
    print("Training Complete")

def main():
    parser = train_parser()
    config = vars(parser.parse_args())
    train(config)

if __name__ == '__main__':
    main()
