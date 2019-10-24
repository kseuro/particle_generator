###############################################################################
# train.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.15.2019
# Purpose: - This file provides a model agnostic training routine. Training is
#            carried out as standard batch-to-batch training for both the
#            GAN and VAE. Each model's respective training function is loaded
#            at runtime from train_fns.py
###############################################################################

# Imports
import os
import torch

# My stuff
import utils
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
    # May leave option open to model parallelization later
    if (config['model'] == 'GAN'):
        g_kwargs, d_kwargs = utils.gan_kwargs(config)
        G = model.Generator(**g_kwargs).to(config['device'])
        D = model.Discriminator(**d_kwargs).to(config['device'])
        model_params = {'g_params': G.parameters(),
                        'd_params': D.parameters()}
        G_optim, D_optim = utils.get_optim(config, model_params)
    elif (config['model'] == 'AE'):
        enc_kwargs, dec_kwargs = utils.ae_kwargs(config)
        E = model.Encoder(**enc_kwargs).to(config['device'])
        D = model.Encoder(**dec_kwargs).to(config['device'])
        model_params = {'e_params': E.parameters(),
                        'd_params': D.parameters()}
        E_optim, D_optim = utils.get_optim(config, model_params)
    elif (config['model'] == 'EWM'):
        emw_kwargs = utils.ewm_kwargs(config)
        G = model.Generator(emw_kwargs).to(config['device'])
        model_params = {'g_params': G.parameters()}
        G_optim = utils.get_optim(config, model_params)
    else:
        raise Exception("No model selected!")

    # Check if resuming from checkpoint
    if config['checkpoint']:
        '''
            This functionality can be written after state_dict saving
            conventions have been established for all three model training
            routines.
        '''
        pass

    # Create dataloader object - either batch-to-batch or full
    if (config['model'] != 'EWM'):
        dataloader = utils.get_LArCV_dataloader(config)
    else:
        dataloader = utils.get_full_dataloader(config)

    # Create optimizer parameter dictionary
    # Update config with D in_features based on image size

    # Generate key word arguments dict for training function
    # G, D, G_D, z_fixed, loss_fn

    # Setup training function (pass config as well)

    # Train model for specified number of epochs

    return 0

def main():
    parser = train_parser()
    config = vars(parser.parse_args())
    train(config)

if __name__ == '__main__':
    main()
