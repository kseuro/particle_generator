# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

#########################
# FullyConnected Layers #
#########################
def FullyConnected(in_f, out_f):
    '''
        Fully connected layers used by both G and D
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.LeakyReLU(0.5)
    )

def G_out(in_f, out_f):
    '''
        Output layer of the generator model
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Tanh()
    )

def G_out_no_actv(in_f, out_f):
    '''
        Output layer of the ewm generator model without tanh()
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
    )

def D_out(in_f, out_f):
    '''
        Output layer of the discriminator model
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Sigmoid()
    )

########################
# Convolutional Layers #
########################
def ConvBlock(in_f, out_f):
    '''
        Convolutional blocks increase the depth of the feature maps from
        in_f -> out_f. The MaxPool2d funtion then reduces the feature
        map dimension by a factor of 2.
    '''
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, kernel_size = 3, padding = 1),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(2,2)
    )

def DeconvBlock(in_f, out_f):
    '''
        Deconvolution function that replaces the usual convolutional transpose
        operation with two linear operations - a bilinear upsample and a
        standard convolution. We set the stride of the convolution operation to
        1 in order to maintain the image dimension after upsampling.
    '''
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(0.2)
    )

def DeconvBlockLast(in_f, out_f):
    '''
        ConvTranspose blocks decrease the depth of the feature maps from
        in_f -> out_f.
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(in_f, out_f, 2, stride = 2),
        nn.Tanh()
    )
