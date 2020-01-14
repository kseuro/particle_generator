import torch.nn as nn
import utils
import train_fns

#####################
# GAN Functionality #
#####################
def get_gan_kwargs(config):
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

def gan(model, config):
    '''
        GAN setup function
    '''
    # Get G and D kwargs based on command line inputs
    g_kwargs, d_kwargs = get_gan_kwargs(config)

    # Set up models on GPU
    G = model.Generator(**g_kwargs).to(config['gpu'])
    D = model.Discriminator(**d_kwargs).to(config['gpu'])

    print(G)
    print(D)
    input('Press any key to launch')

    # Initialize model weights
    G.weights_init()
    D.weights_init()

    # Set up model optimizer functions
    model_params = {'g_params': G.parameters(),
                    'd_params': D.parameters()}
    G_optim, D_optim = utils.get_optim(config, model_params)

    # Set up loss function
    if 'bce' in config['loss_fn']:
        loss_fn = nn.BCELoss().to(config['gpu'])
    else:
        raise Exception("No GAN loss function selected ... aborting")

    # Set up training function
    train_fn = train_fns.GAN_train_fn(G, D, G_optim, D_optim, loss_fn,
                                       config, G_D=None)
    return {'G':G, 'G_optim':G_optim, 'D':D, 'D_optim':D_optim, 'train_fn':train_fn}

#####################
# AE  Functionality #
#####################
def get_ae_kwargs(config):
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

def ae(model, config):
    '''
        AutoEncoder setup function
    '''
    # Get model kwargs
    ae_kwargs, config = get_ae_kwargs(config)

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
    if 'mse' in config['loss_fn']:
        loss_fn = nn.MSELoss().to(config['gpu'])
    elif 'bce' in config['loss_fn']:
        loss_fn = nn.BCELoss().to(config['gpu'])
    else:
        raise Exception("No AutoEncoder loss function selected ... aborting")

    # Set up training function
    train_fn = train_fns.AE_train_fn(AE, AE_optim, loss_fn, config)

    # Return model, optimizer, and model training function
    return {'AE':AE, 'AE_optim':AE_optim, 'train_fn':train_fn}

#####################
# EWM Functionality #
#####################
def ewm_kwargs(config):
    '''
        Create a dictionary of key word arguments for a generator model
        that will be trained to replicate a set of code_vector targets

        - 'dataset' key corresponds to the integer size of the
          code_vector dimension.
    '''
    ewm_kwargs = {}
    if config['MNIST']:
        raise Exception("EWM model is not set up to train on MNIST data")
    code_size = config['l_dim']

    # Creat list of sizes corresponding to the individual
    # fully connected layers in the model
    # e.g. n_hidden = 10, nlayers = 4, fc_sizes = [10,10,10,10]
    fc_sizes = [config['n_hidden']] * config['n_layers']

    ewm_kwargs.update({ 'z_dim'      : config['z_dim'],
                        'fc_sizes'   : fc_sizes,
                        'n_out'      : code_size })
    return ewm_kwargs
