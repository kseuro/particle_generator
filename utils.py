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
from pandas import DataFrame

# My stuff
from dataloader import LArCV_loader
from scipy.ndimage.measurements import center_of_mass as CoM

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
    label = 'MNIST' if config['MNIST'] else 'LArCV'
    config['exp_label'] += '_{}_{}_dataset'.format(label, config['dataset'])

    assert config['save_root'], "No save_root specified in config!"

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

def train_logger(history, best_stats, metrics):
    '''
        Function for tracking training metrics. Determines with each update
        to the training history if that update represents the best model
        performance.
        Args: history (dict): dictionary of training history as lists of floats
              best_stats (dict): dictionary of best loss values
              metrics (dict): most recent loss values as three floats
        Does: updates history dict with most recent metrics. Checks if
              that update is the best yet.
        Returns: history, best_stats, True/False
    '''
    # Check if history is empty before appending data
    # Append most recent training history to loss lists
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
                                                                checkpoint['iter'],
                                                                checkpoint['epoch'])
    torch.save(checkpoint, filename)

def save_sample(sample, epoch, iter, save_dir):
    '''
        - Function for saving periodic samples from the Generator
          function using either with a fixed or random noise vector.
        - Function also saves periodic samples from AutoEncoder
    '''
    # Un-normalize the sample and boost ADC values for better viz.
    # NOTE: This transformation is (should be) un-done when deploy
    #       samples are loaded for analysis.
    sample = ((sample * 0.5) + 0.5) * 10

    if 'fixed' in save_dir:
        im_out = save_dir + 'fixed_sample_{}.png'.format(epoch)
        nrow = 2
        torchvision.utils.save_image(sample[0], im_out, nrow = nrow)
    else:
        im_out = save_dir + 'random_sample_{}_{}.png'.format(epoch, iter)
        nrow = (sample.size(0)//4) if (sample.size(0) % 4) == 0 else 2
        torchvision.utils.save_image(sample, im_out, nrow = nrow)

def shrink_lists(dict):
    '''
        Function that ensures lists in stats dicts are the same length
        for saving with pandas.
    '''
    idx = []
    for key in dict:
        idx.append(len(dict[key]))
    idx = min(idx)
    for key in dict:
        dict[key] = [dict[key][i] for i in range(idx)]
    return dict

def save_train_hist(history, best_stats, times, config, histogram=None):
    '''
        Function for saving network training history and
        best performance stats.
        Args: history (dict): dictionary of network training metrics
              best_stats (dict): dictionary of floating point numbers
                                 representing the best network performance
                                 (i.e. lowest loss)
              times (dict): dictionary of lists containing the training times
              histogram (dict, optional): If training model using EWM algo,
                                          training will produce a dict of
                                          histogram values representing the probability
                                          density distribution of the generator function.
    '''
    # Save times - arrays must all be the same length, otherwise
    # Pandas will thrown an error!
    times_csv = config['save_dir'] + '/times.csv'
    DataFrame(shrink_lists(times)).to_csv(times_csv, header=True, index=False)

    # Save losses
    loss_csv = config['save_dir'] + '/losses.csv'
    DataFrame(shrink_lists(history)).to_csv(loss_csv, header=True, index=False)

    # Save histogram if using EWM algorithm
    if histogram is not None:
        hist_csv = config['save_dir'] + '/histogram.csv'
        DataFrame(histogram).to_csv(hist_csv, header=True, index=False)

    # Save config dict for reference
    df = DataFrame.from_dict(config, orient='index')
    df.to_csv(config['save_dir'] + '/config.csv')

#################################
# Optimizer selection functions #
#################################
def get_optim(config, model_params):
    if (config['model'] == 'gan'):
        return gan_optim(config, model_params)
    elif (config['model'] == 'ae'):
        return ae_optim(config, model_params)
    elif (config['model'] == 'ewm'):
        return ewm_optim(config, model_params)

def gan_optim(config, model_params):
    # G Optimizer
    if ('adam' in config['g_opt']):
        g_optim = torch.optim.Adam(model_params['g_params'], lr=config['g_lr'])
    elif ('sgd' in config['g_opt']):
        g_optim = torch.optim.SGD(model_params['g_params'], lr=config['g_lr'],
                                  momentum=config['p'])
    else:
        raise Exception('G optimizer not selected!')
    # D optimizer
    if ('adam' in config['d_opt']):
        d_optim = torch.optim.Adam(model_params['d_params'], lr=config['d_lr'])
    elif ('sgd' in config['d_opt']):
        d_optim = torch.optim.SGD(model_params['d_params'], lr=config['d_lr'],
                                  momentum=config['p'])
    else:
        raise Exception('D optimizer not selected!')

    return g_optim, d_optim

def ae_optim(config, model_params):
    if ('adam' in config['ae_opt']):
        return torch.optim.Adam(model_params['ae_params'], lr = config['ae_lr'],
                                weight_decay = 1e-5)
    elif ('sgd' in config['ae_opt']):
        return torch.optim.SGD(model_params['ae_params'], lr=config['ae_lr'],
                               momentum=config['p'])
    else:
        raise Exception('AE optimizer not selected!')

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
                          'batch_size' : config['batch_size'],
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
    elif (config['dataset'] == 32):
        config['data_root'] += 'larcv_png_32/'
    else:
        raise Exception('Dataset not specified -- unable to set data_root')
    return config

def MNIST(config):
    transform = transforms.Compose( [transforms.ToTensor(),
                                     transforms.Normalize([0.5],[0.5])])
    data = datasets.MNIST(root='./data', train=True, download=True,
                          transform=transform)
    dataloader = DataLoader(data, **get_loader_kwargs(config))
    return dataloader

def get_LArCV_dataloader(config, loader_kwargs=None):
    '''
        Function that centralizes the setup of the LArCV dataloader.
    '''
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],[0.5])])
    # Select the appropriate dataset
    config = select_dataset(config)

    train_dataset = LArCV_loader(root=config['data_root'], transforms=train_transform)
    if loader_kwargs is None:
        dataloader = DataLoader(train_dataset, **(get_loader_kwargs(config)))
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

def get_dataloader(config):
    if (config['MNIST']):
        return MNIST(config)
    elif (config['model'] != 'ewm'):
        return get_LArCV_dataloader(config)
    else:
        return get_full_dataloader(config)

#####################
# GAN Functionality #
#####################
def gan_kwargs(config):
    '''
        Create two dictionaries of key word arguments for
        generator and discriminator model from config dict.

        - 'dataset' key corresponds to the integer size of one
          data image dimension. i.e. 64 corresponds to the LArCV_PNG_64
          dataset
    '''
    g_kwargs, d_kwargs = {}, {}
    if (config['MNIST']):
        config['dataset'] = 28
    im_size  = config['dataset']**2

    # Creat list of sizes corresponding to the individual
    # fully connected layers in the model(s)
    # e.g. n_hidden = 10, nlayers = 4, fc_sizes = [10,10,10,10]
    fc_sizes = [config['n_hidden']] * config['n_layers']

    g_kwargs.update({ 'z_dim'      : config['z_dim'],
                      'fc_sizes'   : fc_sizes,
                      'n_out'      : im_size})
    d_kwargs.update({'in_features' : im_size,
                     'fc_sizes'    : fc_sizes})
    return g_kwargs, d_kwargs

####################
# AE Functionality #
####################
def ae_kwargs(config):
    kwargs = {}

    # Check if MNIST - set image size
    if (config['MNIST']):
        config['dataset'] = 28
    im_size = config['dataset']**2        # Input dimension
    base = [128 if im_size <= 784 else 256] # Layer base dimension
    l_dim = config['l_dim']               # Latent vector dimension

    # Compute encoder sizes
    # Example output structure: [32*1, 32*2, ... , 32*(2^(n-1))]
    sizes = lambda: [ (yield 2**i) for i in range(config['n_layers']) ]
    enc_sizes = base * config['n_layers']
    enc_sizes = [a*b for a,b in zip(enc_sizes, [*sizes()])][::-1]

    # Update kwarg dicts
    # Decoder is the reverse of the encoder
    kwargs.update({'enc_sizes' : enc_sizes,
                   'l_dim'     : l_dim,
                   'im_size'   : im_size,
                   'dec_sizes' : enc_sizes[::-1]})
    return kwargs, config

#####################
# EWM Functionality #
#####################
def ewm_kwargs(config):  # TODO: Write this function!
    ewm_kwargs = {}
    return ewm_kwargs
