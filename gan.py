################################################################################
# gan.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.08.2019
# Purpose: - This file provides the PyTorch versions of the generator and
#            discriminator network models as multi-layer perceptrons
#          - Model architecture is decided at run-time based on the
#            provided command line arguments and is dynamically allocated
################################################################################

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

############################
# Layer creation functions #
############################
def FullyConnected(in_f, out_f):
    '''
        Fully connected layers used by both G and D
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.LeakyReLU(0.2)
    )

def G_out(in_f, out_f):
    '''
        Output layer of the generator model
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Tanh()
    )

def D_out(in_f, out_f):
    '''
        Output layer of the discriminator model
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Sigmoid()
    )
############################
#       Model Classes      #
############################
class Print(nn.Module):
    '''
        Outputs the shape of convolutional layers in model.
        Call Print() inbetween layers to get shape output to
            the terminal.
    '''
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class Generator(nn.Module):
    '''
        Generator model class
        Does: initializes a G model using the provided arch specs.
        Args: - z_dim (int): dimension of the input vector
              - fc_sizes (list): list of layers sizes to add to model
                - example: fc_sizes = [32, 64, 128] is a 3 layer model
                           with sizes 32, 64, and 128. 
              - n_out (int): dimension of model output layer. Should correspond
                       to the square of the input image dimension
                       - (e.g. [512x512] = 262144)
        Returns: Generator model
    '''
    def __init__(self, z_dim, fc_sizes, n_out):
        super(Generator, self).__init__()
        # Set size of input layer and unpack middle layer sizes as list
        self.fc_sizes = [z_dim, *fc_sizes]
        self.fc = nn.Sequential(*[FullyConnected(in_f, out_f)
                                  for in_f, out_f in zip(self.fc_sizes, self.fc_sizes[1:])])
        ## Output layer
        self.out = G_out(self.fc_sizes[-1], n_out)

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
        return self.out(self.fc(x))

class Discriminator(nn.Module):
    '''
        Discriminator model class
        Does: initializes a D model using the provided arch specs.
        Args: - in_features (int): dimension of the input layer, calculated based
                                   on the dimension of the input data images
              - fc_sizes (list): list of layers sizes to add to model
                - example: fc_sizes = [32, 64, 128] is a 3 layer model
                           with sizes 32, 64, and 128. 
        Returns: Probability of model input belonging to real image space
    '''
    def __init__(self, in_features, fc_sizes):
        super(Discriminator, self).__init__()
        self.fc_sizes = [in_features, *fc_sizes]
        self.fc = nn.Sequential(*[FullyConnected(in_f, out_f)
                                  for in_f, out_f in zip(self.fc_sizes, self.fc_sizes[1:])])
        self.out = D_out(self.fc_sizes[-1], 1)

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
        return self.out(self.fc(x))
