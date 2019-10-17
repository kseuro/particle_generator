################################################################################
# gan.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.08.2019
# Purpose: - This file provides the PyTorch versions of the generator and
#            discriminator network models as multi-layer perceptrons
#          - Model architecture is decided at run-time based on the
#            provided command line arguments,
################################################################################

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

# TODO: add optional dropout functionality
# if self.dropout:
#   self.dropout = nn.Dropout(0.3)
# add dropout to sequential layers

class Generator(nn.Module):
    '''
        Generator model class
        Does: initializes a G model using the provided arch specs. 
        Args: - z_dim (int): dimension of the input vector
              - n_layers (int): number of layers in Generator model   
              - n_hidden (int): number of hidden units in each layer
              - n_out (int): dimension of model output layer. Should correspond
                       to the square of the input image dimension 
                       - (e.g. [512x512] = 262144)
              - optimizer (string): string specifying G's optimizer (adam, sgd)
              - optim_kwargs (dict): dict of optimizer parameters. See docs for
                                     parameter details: 
                                     https://pytorch.org/docs/stable/optim.html
        Returns: Generator model
    '''
    def __init__(self, z_dim, n_layers, n_hidden, n_out, optimizer, optim_kwargs):
        super(Generator, self).__init__()
        self.in_features  = z_dim
        self.n_layers     = n_layers
        self.n_hidden     = n_hidden
        self.out_features = n_out
        self.activation   = nn.LeakyReLU(0.2)
        self.out_activ    = nn.Tanh()
        self.optimizer    = optimizer
        self.optim_kwargs = optim_kwargs

        # Prepare model
        ## First layers
        self.front = nn.Sequential(nn.Linear(self.in_features, self.n_hidden),
                                             self.activation)

        ## Middle layers
        self.layers = []
        for _ in range(self.n_layers - 2):
            self.layers += nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden)
                                                  ,self.activation)
                                                  
        ## Convert self.layers to ModuleList so that it registers properly
        self.layers = nn.ModuleList([nn.ModuleList(layer) for layer in self.layers])

        ## Output layer
        self.out = nn.Sequential(nn.Linear(self.n_hidden, self.out_features), 
                                           self.out_activ)
    
        # Set up optimizer
        if 'adam' in self.optimizer:
            self.optimizer = optim.Adam(**self.optim_kwargs)
        elif 'sgd' in self.optimizer:
            self.optimizer = optim.SGD(**self.optim_kwargs)
        else:
            raise RuntimeError('Optimizer not specified!')
    
    # Initialize the weights
    def weights_init(self):
        '''
            Does: performs normalized initialization of model weights
            Args: None
            Returns: Nothing
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero()

    def forward(self, x):
        x = self.front(x)
        # Loop over dynamically allocated layers
        for _ , layer in enumerate(self.layers):
            x = layer(x)
        return self.out(x)

class Discriminator(nn.Module):
    '''
        Discriminator model class
        Does: initializes a D model using the provided arch specs.
        Args: - in_features (int): dimension of the input layer, calculated based 
                                   on the dimension of the input data images
              - n_layers (int): number of layers in Discriminator model   
              - n_hidden (int): number of hidden units in each layer
              - optimizer (string): string specifying G's optimizer (adam, sgd)
              - optim_kwargs (dict): dict of optimizer parameters. See docs for
                                     parameter details: 
                                     https://pytorch.org/docs/stable/optim.html
        Returns: Probability of model input belonging to real image space
    '''
    def __init__(self, in_features, n_layers, n_hidden, optimizer, optim_kwargs):
        super(Discriminator, self).__init__()
        self.in_features  = in_features
        self.n_layers     = n_layers
        self.n_hidden     = n_hidden
        self.activation   = nn.LeakyReLU(0.2)
        self.dropout      = nn.Dropout(0.3)
        self.out_features = 1
        self.out_activ    = nn.Sigmoid()
        self.optimizer    = optimizer
        self.optim_kwargs = optim_kwargs

        # Prepare model
        ## First layer
        self.front = nn.Sequential(nn.Linear(self.in_features, self.n_hidden),
                                             self.activation)

        ## Middle layers
        self.layers = []
        for _ in range(self.n_layers - 2):
            self.layers += nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden)
                                                  ,self.activation)

        ## Convert self.layers to ModuleList so that it registers properly
        self.layers = nn.ModuleList([nn.ModuleList(layer)
                                     for layer in self.layers])

        ## Output layer
        self.out = nn.Sequential(nn.Linear(self.n_hidden, self.out_features),
                                           self.out_activ)

        # Set up optimizer
        if 'adam' in self.optimizer:
            self.optimizer = optim.Adam(**self.optim_kwargs)
        elif 'sgd' in self.optimizer:
            self.optimizer = optim.SGD(**self.optim_kwargs)
        else:
            raise RuntimeError('Optimizer not specified!')
    
    # Initialize the weights
    def weights_init(self):
        '''
            Does: performs normalized initialization of model weights
            Args: None
            Returns: Nothing
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero()

    def forward(self, x):
        x = self.front(x)
        # Loop over dynamically allocated layers
        for _ , layer in enumerate(self.layers):
            x = layer(x)
        return self.out(x)
