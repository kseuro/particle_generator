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
    if (config['model'] == 'GAN'):
        # Get G and D kwargs based on command line inputs
        g_kwargs, d_kwargs = utils.gan_kwargs(config)
        
        # Set up models on GPU
        G = model.Generator(**g_kwargs).to(config['device'])
        D = model.Discriminator(**d_kwargs).to(config['device'])
        
        # Set up model optimizer functions
        model_params = {'g_params': G.parameters(),
                        'd_params': D.parameters()}
        G_optim, D_optim = utils.get_optim(config, model_params)
        
        # Set up loss function
        loss_fn = nn.BCELoss().to(config['device'])

        # Set up training function
        train_GAN = train_fns.GAN_train_fn(G, D, G_D=None, G_optim, D_optim, 
                                           loss_fn, config)
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
        # Set up fixed noise vector
        z_fixed = torch.randn(config['batch_size'],
                              config['z_dim'], device=config['device'])
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

    # Create dataloader object - either batch-to-batch, full, or MNSIT
    if (config['MNIST']):
        dataloader = utils.MNIST(config)
    elif (config['model'] != 'EWM'):
        dataloader = utils.get_LArCV_dataloader(config)
    else:
        dataloader = utils.get_full_dataloader(config)

    # Set up progress bar for terminal output and enumaeration
    prog_bar = tqdm(dataloader)

    # Empty dict for tracking training metrics
    history = {}

    # Set up directories for saving training stats and outputs
    config = utils.directories(config)

    # Train model for specified number of epochs
    for epoch in range(config['num_epochs']):
        # TODO: Set up training loop for MNIST data
        for i, (x, _) in enumerate(prog_bar):
            metrics = train_GAN(x)
            history = utils.train_logger(history, metrics)

            # Save periodically
            # TODO: Write function that saves training history
    
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
