############################################################################
# gan.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.08.2019
# Purpose: - This file provides the PyTorch versions of the generator and
#            discriminator network models as multi-layer perceptrons
#          - Model architecture is decided at run-time based on the
#            provided command line arguments,
############################################################################

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

class Generator(nn.Module):
    '''
        Generator model class
        Does: initializes a G model using the provided arch specs. 
        Args: - z_dim: dimension of the input vector
              - n_layers: number of layers in Generator model   
              - n_hidden: number of hidden units in each layer
              - n_out: dimension of model output layer. Should correspond
                       to the square of the input image dimension 
                       - (e.g. [512x512] = 262144)
        Returns: Generator model
    '''
    def __init__(self, z_dim, n_layers, n_hidden, n_out):
        super(Generator, self).__init__()
        self.in_features  = z_dim
        self.n_layers     = n_layers
        self.n_hidden     = n_hidden
        self.out_features = n_out
        self.activation   = nn.LeakyReLU(0.2)
        self.out_activ    = nn.Tanh()

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
        Args: - n_features: dimension of the input layer, calculated based on                       
                            the dimension of the input data images
              - n_layers: number of layers in Discriminator model   
              - n_hidden: number of hidden units in each layer
        Returns: Probability of model input belonging to real image space
    '''
    def __init__(self, in_features, n_layers, n_hidden):
        super(Discriminator, self).__init__()
        self.in_features = in_features
        self.n_layers    = n_layers
        self.n_hidden    = n_hidden
        self.activation  = nn.LeakyReLU(0.2)
        self.dropout     = nn.Dropout(0.3)
        self.out_activ   = nn.Sigmoid()
        self.n_out       = 1

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

    def forward(self, x):
        x = self.front(x)
        # Loop over dynamically allocated layers
        for _ , layer in enumerate(self.layers):
            x = layer(x)
        return self.out(x)


