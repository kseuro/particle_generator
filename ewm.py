################################################################################
# ewm.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.23.2019
# Purpose: - This file provides a PyTorch version if a generator network to
#            be trained using the EWM algorithm
#          - Model architecture is decided at run-time based on the
#            provided command line arguments,
################################################################################
import torch
import torch.nn as nn
import torchvision.utils

from gan import Print
from gan import FullyConnected

############################
# Layer creation functions #
############################
def G_out(in_f, out_f):
    '''
        Output layer of the ewm generator model - does not include tanh()
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
    )

class ewm_G(nn.Module):
    '''
        Per Chen et. al (2019), the MLP generator has the following
        architecture:
            Input(100) -> FC -> Hidden(512) -> FC -> Hidden(512)
                       -> FC -> Hidden(512) -> Output(D)
            Where D is the dimension of the output image.

        Here, EWM generator model is the same as the model used in the
        MLP GAN, but with an unbounded activation function at the output layer.
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
                m.bias.data.zero_()

    def forward(self, x):
        return self.out(self.fc(x))
