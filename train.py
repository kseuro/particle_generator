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
        Args: config (dictionary): config dict of parsed command line params.
        Returns: Nothing
    '''
    # Import the selected model
    model = __import__(config['model']) # GAN, AE, or EWM

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
        print(type(config))
        print(type(model_params))
        G_optim, D_optim = utils.get_optim(config, model_params)

        # Set up loss function
        loss_fn = nn.BCELoss().to(config['gpu'])

        # Set up training function
        train_fn = train_fns.GAN_train_fn(G, D, G_optim, D_optim, loss_fn,
                                           config, G_D=None)
    elif (config['model'] == 'ae'):
        enc_kwargs, dec_kwargs = utils.ae_kwargs(config)
        E = model.Encoder(**enc_kwargs).to(config['gpu'])
        D = model.Encoder(**dec_kwargs).to(config['gpu'])
        model_params = {'e_params': E.parameters(),
                        'd_params': D.parameters()}
        E_optim, D_optim = utils.get_optim(config, model_params)
    elif (config['model'] == 'ewm'):
        emw_kwargs = utils.ewm_kwargs(config)
        G = model.Generator(emw_kwargs).to(config['gpu'])
        model_params = {'g_params': G.parameters()}
        G_optim = utils.get_optim(config, model_params)
        # Set up fixed noise vector
        z_fixed = torch.randn(config['batch_size'],
                              config['z_dim'], gpu=config['gpu'])
    else:
        raise Exception("No model selected!")

    # Check if resuming from checkpoint
    if config['checkpoint']:
        '''
            This functionality should be written after state_dict saving
            conventions have been established for all three model training
            routines.
        '''
        pass

    # Create dataloader object - either batch-to-batch, full, or MNSIT
    if (config['MNIST']):
        dataloader = utils.MNIST(config) # Get MNIST data (DL if not available)
    elif (config['model'] != 'ewm'):
        dataloader = utils.get_LArCV_dataloader(config)
    else:
        # TODO: This may need to be modified after AE training is complete
        dataloader = utils.get_full_dataloader(config)

    # Set up progress bar for terminal output and enumeration
    prog_bar = tqdm(dataloader)

    # Empty dicts for tracking training metrics and best stats
    history, best_stats = {}, {}

    # Set up directories for saving training stats and outputs
    config = utils.directories(config)

    # Train model for specified number of epochs
    for epoch in range(config['num_epochs']):
        # MNIST training loop
        for itr, (x, _) in enumerate(prog_bar):
            metrics = train_fn(x)
            history, best_stats, best = utils.train_logger(history,
                                                           best_stats,
                                                           metrics)
            # Save checkpoint periodically
            if (itr % 2000 == 0):
                 chkpt_G = utils.get_checkpoint(itr, epoch, G, G_optim)
                 utils.save_checkpoint(chkpt_G, best, 'G', config['weights_save'])
                 chkpt_D = utils.get_checkpoint(itr, epoch, D, D_optim)
                 utils.save_checkpoint(chkpt_D, best, 'D', config['weights_save'])

            # Save Generator output periodically
            if (itr % 1000 == 0):
                z_rand = torch.randn(config['sample_size'],
                                     config['z_dim'], gpu=config['gpu'])
                sample = G(z_rand).view(-1, 1, config['dataset'], config['dataset'])
                utils.save_sample(sample, epoch, itr, config['random_samples'])

        # Save Generator output using fixed vector at end of epoch
        sample = G(z_fixed).view(-1, 1, config['dataset'], config['dataset'])
        utils.save_sample(sample, epoch, itr, config['fixed_samples'])

    # Save training history and model architecture for evaluation and deploy


        # TODO: Set up training loop for LArCV data
        # for i, x in enumerate(prog_bar):
        # train


    # Save and process training history
    # TODO: Write function that processes the history dict
    print("Training Complete")

    return

def main():
    parser = train_parser()
    config = vars(parser.parse_args())
    train(config)

if __name__ == '__main__':
    main()
