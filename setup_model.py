import torch.nn as nn
import utils

def gan(model, config):
    '''
        GAN setup function
    '''
    # Get G and D kwargs based on command line inputs
    g_kwargs, d_kwargs = utils.gan_kwargs(config)

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
    loss_fn = nn.BCELoss().to(config['gpu'])

    # Set up training function
    train_fn = train_fns.GAN_train_fn(G, D, G_optim, D_optim, loss_fn,
                                       config, G_D=None)
    return {'G':G, 'G_optim':G_optim, 'D':D, 'D_optim':D_optim,
            'train_fn':train_fn, 'loss_fn':loss_fn}

def ae(model, config):
    '''
        AutoEncoder setup function
    '''
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
    return {'AE':AE, 'AE_optim':AE_optim, 'train_fn':train_fn, 'loss_fn':loss_fn}
