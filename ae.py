################################################################################
# ae.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.23.2019
# Purpose: - This file provides the PyTorch versions of the encoder and
#            decoder network models for use in the particle generator
#          - Model architecture is decided at run-time based on the
#            provided command line arguments.
################################################################################

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from ae_layers import *

class Encoder(nn.Module):
    def __init__(self, enc_sizes, l_dim):
        super().__init__()
        self.fc_blocks = nn.Sequential(*[enc_block(in_f, out_f) for in_f, out_f
                                        in zip(enc_sizes, enc_sizes[1:])])
        self.last = nn.Linear(enc_sizes[-1], l_dim)

    # Initialize the weights
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        return self.last(self.fc_blocks(x))

class Decoder(nn.Module):
    def __init__(self, dec_sizes, im_size):
        super().__init__()
        self.fc_blocks = nn.Sequential(*[dec_block(in_f, out_f) for in_f, out_f
                                        in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Sequential(nn.Linear(dec_sizes[-1], im_size), nn.Tanh())

    # Initialize the weights
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        return self.last(self.fc_blocks(x))

class AutoEncoder(nn.Module):
    def __init__(self, enc_sizes, l_dim, dec_sizes, im_size):
        super().__init__()
        self.enc_sizes = [im_size] + [*enc_sizes]
        self.dec_sizes = [l_dim] + [*dec_sizes]
        self.encoder = Encoder(self.enc_sizes, l_dim)
        self.decoder = Decoder(self.dec_sizes, im_size)

    def weights_init(self):
        self.encoder.weights_init()
        self.decoder.weights_init()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvEncoder(nn.Module):
    '''
        Convolutional AutoEncoder - Encoder branch
        args: depth (int list): list of integer feature depth sizes
                                that define the output volume of the
                                individual convolution operations.
                                The depth list is computed by dividing
                                the desired depth by the number of layers.
                                Ex: D = 32, n_layers = 4, depth = [4, 8, 16, 32]
                                Adding the initial channel: [1, 4, 8, 16, 32]
    '''
    def __init__(self, depth):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f) for in_f, out_f
                                           in zip(depth, depth[1:])])
    def forward(self, x):
        return self.conv_blocks(x)

class ConvDecoder(nn.Module):
    '''
        Convolutional AutoEncoder - Decoder branch
        args: depth (int list): list of integer feature depth sizes
                                that define the output volume of the
                                individual transposed convolution operations.
                                See ConvEncoder for example of list, which is
                                reversed in the Decoder model.
    '''
    def __init__(self, depth):
        super().__init__()
        self.deconv_blocks = nn.Sequential(*[deconv_blocks(in_f, out_f) for in_f, out_f
                                              in zip(depth, depth[1:])])
        self.activation = nn.Tanh()
    def forward(self, x):
        return self.activation(self.deconv_blocks(x))

class ConvAutoEncoder(nn.Module):
    '''
        Convolutional AutoEncoder model
    '''
    def __init__(self, depth):                 # Examples:
        self.enc_features = [1] + depth        # [1, 4, 8, 16, 32]
        self.dec_features = depth[::-1] + [1]  # [32, 16, 8, 4, 1]
        self.encoder = ConvEncoder(self.enc_features)
        self.decoder = ConvDecoder(self.dec_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
