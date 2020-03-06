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
from gan import DeconvBlock
from gan import DeconvBlockLast

############################
# Weight initialization Fn #
############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
        super(ewm_G, self).__init__()
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
        super(ewm_G, self).__init__()
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

class ewm_convG(nn.Module):
    """
    Designed to map a latent space vector (z) to data-space. Since the data
        are images, the conversion of z to data-space means creating an image
        with the same size as the training images (1x28x28).
    In practice, this is done with a series of strided 2D conv-transpose
        layers, paired with a 2D batch-norm layer and ReLU activation.
        The output is passed through a Tanh function to map it to the input
        data range of [-1, 1].
    Inputs:
        - nc:  number of color channels    (rgb = 3) (bw = 1)
        - nz:  length of the latent vector (100)
        - ngf: depth of feature maps carried through generator
        - Transposed convolution is also known as fractionally-strided conv.
            - One-to-many operation
    ConvTranspose2d output volume:
        Input:  [N, C, Hin,  Win]
        Output: [N, C, Hout, Wout] where:
            Hout = (Hin - 1) * stride - 2 * pad + K + out_pad (default = 0)
            Wout = (Win - 1) * stride - 2 * pad + K + out_pad (default = 0)
            K = 4, S = 2, P = 1: doubles img. dim each layer
    """
    def __init__(self, l_dim, dec_sizes, im_size):
        super(ewm_convG, self).__init__()
        self.deconv_sizes = [l_dim] + [*dec_sizes]
        self.main = nn.Sequential(*[DeconvBlock(in_f, out_f)
                                  for in_f, out_f in zip(self.deconv_sizes, self.deconv_sizes[1:-1])]
        )
        self.last = DeconvBlockLast(self.deconv_sizes[-2], self.deconv_sizes[-1])
    def forward(self, z_h):
        return self.last(self.main(z_h))
