###############################################################################
# utils.py
# Author: Kai Stewart
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import DataFrame

# My stuff
from dataloader import LArCV_loader
from dataloader import BottleLoader
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

    if config['model'] != 'ewm':
        label = 'MNIST' if config['MNIST'] else 'LArCV'
        if 'ae' in config['model']:
            config['exp_label'] += '_{}_{}_dataset_{}_l-dim/'.format(label,
                                                                    config['dataset'],
                                                                    config['l_dim'])
        else:
            config['exp_label'] += '_{}_{}_dataset/'.format(label, config['dataset'])
    else:
        label = 'Code_Vectors_{}_{}'.format(config['dataset'], config['l_dim'])
        config['exp_label'] += '_{}/'.format(label)

    assert config['save_root'], "No save_root specified in config!"

    # Create path for experiment
    save_dir = config['save_root'] + config['exp_label']
    config.update({'save_dir' : save_dir})
    dirs.append(config['save_dir'])

    # Create path for saving weights
    config.update({'weights_save' : config['save_dir'] + 'weights/'})
    dirs.append(config['weights_save'])

    # Sample saving
    samples_dir = save_dir + 'training_samples/'
    dirs.append(samples_dir)

    # Random samples
    config.update( {'random_samples' : samples_dir + 'random_samples/' } )
    dirs.append(config['random_samples'])

    # Fixed samples
    config.update({'fixed_samples': samples_dir + 'fixed_samples/'})
    dirs.append(config['fixed_samples'])

    # OTS Histograms
    if config['model'] == 'ewm':
        config.update( {'histograms' : config['save_dir'] + 'histograms/'})
        dirs.append(config['histograms'])

    # Make directories for saving
    for i in range(len(dirs)):
        make_dir(dirs[i])

    return config

def train_logger(history, best_stat, metrics):
    '''
        Function for tracking training metrics. Determines, with each update
        to the training history, if that update represents the best model
        performance.
        Args: history (dict): dictionary of training history as lists of floats
              best_stat (dict): dictionary of best loss values
              metrics (dict): most recent loss values as three floats
        Does: updates history dict with most recent metrics.
        Returns: history, best_stat
    '''
    # Check if history is empty before appending data
    # Append most recent training history to loss lists
    if not history:
        for key in metrics:
            history.update( { key: [metrics[key]] } )
    else:
        for key in metrics:
            history[key].append(metrics[key])

    check = []
    # Check if best_stat is empty before appending data
    if not best_stat:
        for key in history:
            best_stat.update( { key: history[key][-1] } )
    else:
        # Compare the last recorded loss value with the current
        # best_stat. If that loss value is lower than the best_stat,
        # then update the best_stat.
        for key in history:
            if round(history[key][-1], 5) < round(best_stat[key], 5):
                best_stat[key] = history[key][-1]
    return history, best_stat

def get_checkpoint(epoch, kwargs, config):
    '''
        Function for generating a model checkpoint dictionary
    '''
    dict = {}
    if 'ae' in config['model']:
        dict.update( { 'epoch'      : epoch,
                       'state_dict' : kwargs['AE'].state_dict(),
                       'optimizer'  : kwargs['AE_optim'].state_dict() } )
    elif 'gan' in config['model']:
        # Write model checkpoint save for 'G' and 'D' in kwargs
        pass
    elif 'ewm' in config['model']:
        dict.update( { 'epoch'      : epoch,
                       'state_dict' : kwargs['G'].state_dict(),
                       'optimizer'  : kwargs['G_optim'].state_dict() } )
    return dict

def save_checkpoint(checkpoint, config):
    '''
        Function for saving model and optimizer weights
        Args: checkpoint (dict): dictionary of model weights
              config (dict): experiment configuration dictionary
              save_dir (str): full path to save location
    '''
    save_dir = config['weights_save']
    chkpt_name = 'best_{}_ep_{}.tar'.format(config['model'], checkpoint['epoch'])
    filename = save_dir + chkpt_name
    torch.save(checkpoint, filename)

def save_sample(sample, epoch, iter, save_dir):
    '''
        - Function for saving periodic samples from the Generator
          function using either with a fixed or random noise vector.
        - Function also saves periodic samples from AutoEncoder
    '''
    if 'fixed' in save_dir:
        im_out = save_dir + 'fixed_sample_{}.png'.format(epoch)
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

def get_arch(config):
    arch = {}
    # Architecture features common to both models
    arch.update( { 'n_layers'   : config['n_layers'],
                   'num_epochs' : config['num_epochs'],
                   'model'      : config['model'] } )
    # Model dependent features
    if 'ae' in config['model']:
        arch.update( { 'l_dim' : config['l_dim']} )
    if 'gan' in config['model']:
        arch.update( { 'z_dim'    : config['z_dim'],
                       'n_hidden' : config['n_hidden'] } )
    if 'ewm' in config['model']:
        arch.update( { 'dataset'  : config['dataset'],
                       'n_hidden' : config['n_hidden'],
                       'l_dim'    : config['l_dim'],
                       'z_dim'    : config['z_dim'] } )
    return arch

def save_train_hist(history, config, times=None, histogram=None):
    '''
        Function for saving network training history and
        best performance stats.
        Args: history (dict): dictionary of network training metrics
              best_stat (dict): dictionary of floating point numbers
                                 representing the best network performance
                                 (i.e. lowest loss)
              times (dict): dictionary of lists containing the training times
              histogram (dict, optional): If training model using EWM algo,
                                          training will produce a dict of
                                          histogram values representing the probability
                                          density distribution of the generator function.
    '''
    # Save times - arrays must all be the same length, otherwise Pandas will thrown an error!
    if times is not None:
        # This is bad coding, but only the gan and ae models save the training
        # times and will therefore supply a list to this conditional.
        times_csv = config['save_dir'] + '/times.csv'
        DataFrame(shrink_lists(times)).to_csv(times_csv, header=True, index=False)
        # Save losses
        loss_csv = config['save_dir'] + '/losses.csv'
        DataFrame(shrink_lists(history)).to_csv(loss_csv, header=True, index=False)
    else:
        ots_loss = config['save_dir'] + '/ots_losses.csv'
        DataFrame(history['losses']['ot_loss']).to_csv(ots_loss, header=True, index=False)
        fit_loss = config['save_dir'] + '/fit_losses.csv'
        DataFrame(history['losses']['fit_loss']).to_csv(fit_loss, header=True, index=False)

    # Save histogram if using EWM algorithm
    # Convert the histogram dict to csv file using Pandas
    if histogram is not None:
        hist_csv = config['save_dir'] + '/histogram.csv'
        df = DataFrame.from_dict(histogram, orient='index')
        df.to_csv(hist_csv)

    # Save config dict for reference
    df = DataFrame.from_dict(config, orient='index')
    df.to_csv(config['save_dir'] + '/config.csv')

    # Save model architecture
    arch = get_arch(config)
    arch_file = config['save_dir'] + '/model_arch.csv'
    DataFrame.from_dict(arch, orient="index").to_csv(arch_file)

#################################
# Optimizer selection functions #
#################################
def get_optim(config, model_params):
    if 'gan' in config['model']:
        return gan_optim(config, model_params)
    elif 'ae' in config['model']:
        return ae_optim(config, model_params)
    elif 'ewm' in config['model']:
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
    if ('adam' in config['g_opt']):
        ewm_optim = torch.optim.Adam(model_params['g_params'], lr=config['g_lr'])
    elif ('sgd' in config['g_opt']):
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

def get_dset_size(data_root):
    return sum( [len(examples) for _, _, examples in os.walk(data_root)] )

def select_dataset(config):
    '''
        Function that appends the appropriate path suffix to the data_root
        based on dataset value. This is necessary because of the folder
        structure expected by the torch ImageFolder class.
    '''
    if config['model'] == 'ewm':
        if config['ewm_target'] == 'conv':
            config['data_root'] += 'conv_ae/'
        else:
            config['data_root'] += 'mlp/'
        config['data_root'] += 'code_vectors_{}_{}/'.format(config['dataset'], config['l_dim'])
        return config
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

def select_test_vecs(config):
    if config['ewm_target'] == 'conv':
        config['vec_root'] += 'conv_ae/'
    elif config['ewm_target'] == 'mlp':
        config['vec_root'] += 'mlp/'
    else:
        raise Exception('EWM Target not specified -- unable to select test vectors')
    config['vec_root'] += "code_vectors_{}_{}/".format(config['dataset'], config['l_dim'])
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
    train_dataset = LArCV_loader(root=config['data_root'], transforms=train_transform)
    if loader_kwargs is None:
        dataloader = DataLoader(train_dataset, **(get_loader_kwargs(config)))
    else:
        dataloader = DataLoader(train_dataset, **loader_kwargs)
    return dataloader

def get_BottleLoader(config, loader_kwargs=None):
    '''
        Function that sets up the loading of code_vector targets.
            -  The code_vectors are .csv files loaded as NumPy arrays
               inside the BottleLoader object. They only need to be
               cast to Torch Tensors, as we wish to perserve their structure.
    '''
    train_transform = transforms.Compose([ transforms.ToTensor() ])
    train_dataset = BottleLoader(root=config['data_root'], transforms=train_transform)
    dataloader = DataLoader(train_dataset, **loader_kwargs)
    return dataloader

def get_full_dataloader(config):
    '''
        Returns a dataloader containing full set of code_vector examples, or full set
        of LArCV_[nxn] images. Be careful not to overload the GPU memory.
    '''
    loader_kwargs = get_loader_kwargs(config)
    loader_kwargs.update({'batch_size': get_dset_size(config['data_root'])})
    if config['model'] == 'ewm_conv':
        dataloader = get_LArCV_dataloader(config, loader_kwargs=loader_kwargs)
    else:
        dataloader = get_BottleLoader(config, loader_kwargs=loader_kwargs)
    for data in dataloader:
        print('Returning full dataloader with {} training examples'.format(loader_kwargs['batch_size']))
        return data

def get_dataloader(config):
    config = select_dataset(config)
    if (config['MNIST']):
        if 'ewm' in config['model']:
            raise Exception("EWM model is not set up to train on MNIST data")
        return MNIST(config)
    elif 'ewm' in config['model']:
        return get_full_dataloader(config) # Train EWM Generator
    else:
        return get_LArCV_dataloader(config) # Train Conv or MLP AE or GAN

def get_test_loader(config):
    config = select_test_vecs(config)
    loader_kwargs = get_loader_kwargs(config)
    train_transform = transforms.Compose([ transforms.ToTensor() ])
    test_dataset = BottleLoader(root=config['vec_root'], transforms=train_transform)
    dataloader = DataLoader(test_dataset, **loader_kwargs)
    return dataloader

#####################
# EWM Functionality #
#####################
def save_histogram(histogram, history, config):
    fig = plt.hist(histogram)
    fig_name = 'OTS_Histogram_{}_{}.png'.format(history['epoch'], history['iter'])
    plt.title(fig_name)
    plt.savefig(config['histograms'] + fig_name)

def update_histogram(transfer, history, config):

    hist = np.histogram(transfer.reshape(-1), bins=1000, range=(0, history['dset_size'] -1 ))[0]
    ots_loss = np.mean(history['losses']['ot_loss'][-1000:])
    print("-"*60)
    print("OTS Epoch {}, iteration {}".format(history['epoch'], history['iter']))
    print("OTS Loss  {:.2f}".format(ots_loss))
    print("Histogram: (min: {}, max {})".format(hist.min(), hist.max()))
    print("-"*60)
    # Update the histogram dict
    history['hist_dict']['hist_min'].append(hist.min())
    history['hist_dict']['hist_max'].append(hist.max())
    history['hist_dict']['ot_loss'].append(ots_loss)

    save_histogram(hist, history, config)

    stop = False
    min_check = hist.min() >= config['early_end'][0]
    max_check = hist.max() <= config['early_end'][1]
    if min_check and max_check:
        stop = True

    return history, stop
