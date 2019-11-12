###############################################################################
# utils.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.16.2019
# Purpose: - This file provides setup functionality for model training,
#            resuming training from checkpoint, metric tracking, and
#            saving outputs.
###############################################################################

# Imports
import os
import torch
import torchvision
from torchvision      import datasets
from torchvision      import transforms
from torch.utils.data import DataLoader
import time
from datetime import datetime

# My stuff
from dataloader import LArCV_loader

#################################
#     Logging Functionality     #
#################################
def make_dir(dir):
    '''
        Simple utility function for creating directories
    '''
    if os.path.exists(dir):
        return
    try:
        os.makedirs(dir)
    except OSError as E:
        if E.errno != errno.EEXIST:
            raise

def train_logger(history, metrics, best_stats):
    '''
        Function for tracking training metrics. Determines with each update
        to the training history if that update represents the best model
        performance.
        Args: history (dict): dictionary of training history as lists of floats
              metrics (dict): most recent loss values as three floats
              best_stats (dict): dictionary of best loss values
        Does: updates history dict with most recent metrics. Checks if
              that update is the best yet.
        Returns: history, best_stats, True/False
    '''
    # Check if history is empty before appending data
    if not history:
        for key in metrics:
            history.update({ key: [metrics[key]] })
    else:
        for key in metrics:
            history[key].append(metrics[key])

    check = []
    # Check if best_stats is empty before appending data
    if not best_stats:
        for key in history:
            best_stats.update({key: history[key][len(history[key]) - 1]})
            check.append(True)
    else:
        for key in history:
            threshold = best_stats[key] - round(best_stats[key] * 0.5, 3)
            if (history[key][len(history[key]) - 1]) < threshold:
                best_stats[key] = history[key][len(history[key]) - 1]
                check.append(True)
            else:
                check.append(False)
    return history, best_stats, all(check)

def directories(config):
    '''
        Function that generates a label for the experiement based on the date,
            time, and training dataset.
        Creates directories for saving weights, samples, and other outputs.
    '''
    dirs = []
    # Date and time labelling
    now  = datetime.now()
    date = now.strftime("%m-%d-%Y")
    time = now.strftime("%H-%M-%S")

    # Create a label for the current experiment
    prefix = '{}_{}_{}'.format(date, time, config['model'])
    config['exp_label'] = prefix + '_{}_epochs'.format(config['num_epochs'])

    assert not config['save_root'], "No save_root specified in config!"

    # Create path for experiment
    save_dir = config['save_root'] + config['exp_label']
    config.update({'save_dir' : save_dir})
    dirs.append(config['save_dir'])

    # Create path for saving weights
    config.update({'weights_save' : config['save_dir'] + '/weights/'})
    dirs.append(config['weights_save'])

    # Sample saving
    dirs.append(save_dir + '/training_samples/')

    # Random samples
    config.update( {'random_samples' : 
                    config['save_dir'] + 
                    '/training_samples/' + 
                    'random_samples/'})
    dirs.append(config['random_samples'])

    # Fixed samples
    config.update({'fixed_samples':
                   config['save_dir'] +
                   '/training_samples/' +
                   'fixed_samples/'})
    dirs.append(config['fixed_samples'])

    # OTS Histograms
    if (config['model'] == 'EWM' or config['model'] == 'ewm'):
        config.update( {'histograms' : config['save_dir'] + '/histograms/'})
        dirs.append(config['histograms'])

    # Make directories for saving
    for i in range(len(dirs)):
        make_dir(dirs[i])
    
    return config

def get_checkpoint(iter, epoch, model, optim):
    '''
        Function for generating a model checkpoint dictionary
    '''
    dict = {}
    dict.update({'iter'      : iter,
                 'epoch'     : epoch,
                 'state_dict': model.state_dict(),
                 'optimizer' : optim.state_dict()
                })
    return dict

def save_checkpoint(checkpoint, best, model_name, save_dir):
    '''
        Function for saving model and optimizer weights
        Args: checkpoint (dict): dictionary of model weights
              best (bool): boolean corresponding to best checkpoint
              save_dir (str): full path to save location
    '''
    if best:
        filename = save_dir + 'best_chkpt_{}_{}.tar'.format(model_name, 
                                                            checkpoint['epoch'])
    else:
        filename = save_dir + 'chkpt_{}_it_{}_ep_{}.tar'.format(model_name, 
                                                                checkpoint['epoch'], 
                                                                checkpoint['iter'])
    torch.save(checkpoint, filename)

def save_sample(sample, epoch, iter, save_dir):
    '''
        Function for saving periodic samples from the Generator
        function, using either with a fixed or random noise vector.
    '''
    # Un-normalize the sample and boost ADC values for better viz.
    sample = ((sample * 0.5) + 0.5) * 10
    if 'fixed' in save_dir:
        im_out = save_dir + 'fixed_sample_{}.png'.format(epoch)
        torchvision.utils.save_image(sample[0], im_out)
    else:
        im_out = save_dir + 'random_sample_{}_{}.png'.format(epoch, iter)
        torchvision.utils.save_image(sample, im_out)

#################################
# Optimizer selection functions #
#################################
def get_optim(config, model_params):
    if (config['model'] == 'GAN'):
        return gan_optim(config, model_params)
    elif (config['model'] == 'AE'):
        return ae_optim(config, model_params)
    elif (config['model'] == 'EWM'):
        return ewm_optim(config, model_params)

def gan_optim(config, model_params):
    # G Optimizer
    if ('adam' in config['g_optim']):
        g_optim = torch.optim.Adam(model_params['g_params'], lr=config['g_lr'])
    elif ('sgd' in config['g_optim']):
        g_optim = torch.optim.SGD(model_params['g_params'], lr=config['g_lr'],
                                  momentum=config['p'])
    else:
        raise Exception('G optimizer not selected!')
    
    # D optimizer
    if ('adam' in config['d_optim']):
        d_optim = torch.optim.Adam(model_params['d_params'], lr=config['d_lr'])
    elif ('sgd' in config['d_optim']):
        d_optim = torch.optim.SGD(model_params['d_params'], lr=config['d_lr'],
                                  momentum=config['p'])
    else:
        raise Exception('D optimizer not selected!')

    return g_optim, d_optim

def ae_optim(config, model_params):
    # Encoder optimizer
    e_optim = 0
    # Decoder optimizer
    d_optim = 0
    return e_optim, d_optim

def ewm_optim(config, model_params):
    # Generator optimizer
    if ('adam' in config['ewm_optim']):
        ewm_optim = torch.optim.Adam(model_params['g_params'], lr=config['g_lr'])
    elif ('sgd' in config['ewm_optim']):
        ewm_optim = torch.optim.SGD(model_params['g_params'], lr=config['g_lr'],
                                    momentum=config['p'])
    else:
        raise Exception('EWM optimizer not selected!')
    
    return ewm_optim

############################
# Dataloader Functionality #
############################
def get_loader_kwargs(config):
    loader_kwargs = {}
    loader_kwargs.update({'num_workers': config['num_workers'],
                          'batch_size' : config['data_limit'],
                          'shuffle'    : config['shuffle'],
                          'drop_last'  : config['drop_last']})
    return loader_kwargs

def select_dataset(config):
    '''
        Function that appends the appropriate path suffix to the data_root
        based on dataset value. This is necessary because of the folder
        structure expected by the torch ImageFolder class.
    '''
    if (config['dataset'] == 512):
        config['data_root'] += 'larcv_png_512/'
    elif (config['dataset'] == 256):
        config['data_root'] += 'larcv_png_256/'
    elif (config['dataset'] == 128):
        config['data_root'] += 'larcv_png_128/'
    elif (config['dataset'] == 64):
        config['data_root'] += 'larcv_png_64/'
    else:
        raise Exception('Dataset not specified -- unable to set data_root')
    return config

def MNIST(config):
    transform = transforms.Compose( [transforms.ToTensor(),
                                     transforms.Normalize((.5, .5, .5),
                                                          (.5, .5, .5))])
    data = datasets.MNIST(root='./data', train=True, download=True, 
                          transform=transform)
    dataloader = DataLoader(data, **get_loader_kwargs(config))
    return dataloader

def get_LArCV_dataloader(config, loader_kwargs=None):
    '''
        Function that centralizes the setup of the dataloader.
    '''
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],[0.5])])
    # Select the appropriate dataset
    config = select_dataset(config)

    train_dataset = LArCV_loader(root=config['data_root'], transforms=train_transform)
    if loader_kwargs is None:
        dataloader = DataLoader(train_dataset, **get_loader_kwargs(config))
    else:
        dataloader = DataLoader(train_dataset, **loader_kwargs)
    return dataloader

def get_full_dataloader(config):
    '''
        Returns a dataloader containing 10000 training images.
        10000 is safe to load onto a Nvidia Titan 1080x with a single
        model also loaded. 20000 may also work, but the operating system may 
        squash the thread at a higher number.
    '''
    loader_kwargs = get_loader_kwargs(config)
    loader_kwargs.update({'batch_size': 10000})
    dataloader = get_LArCV_dataloader(config, loader_kwargs=loader_kwargs)
    for data in dataloader:
        print('Returning full dataloader')
        return data

#####################
# GAN Functionality #
#####################
def gan_kwargs(config):
    g_kwargs, d_kwargs = {}, {}
    if (config['MNIST']):
        config['dataset'] = 28
    g_kwargs.update({ 'z_dim'    : config['z_dim'],
                      'n_layers' : config['n_layers'],
                      'n_hidden' : config['n_hidden'],
                      'n_out'    : config['dataset']**2})
    d_kwargs.update({'in_features': config['dataset']**2,
                     'n_layers'   : config['n_layers'],
                     'n_hidden'   : config['n_hidden']})
    return g_kwargs, d_kwargs

#####################
# EWM Functionality #
#####################
def ewm_kwargs(config): # TODO: Write this function!
    ewm_kwargs = {}
    return ewm_kwargs

####################
# AE Functionality #
####################
def ae_kwargs(config):  # TODO: Write this function!
    enc_kwargs, dec_kwargs = {}, {}
    return enc_kwargs, dec_kwargs