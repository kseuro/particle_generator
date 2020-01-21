# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

def enc_block(in_f, out_f):
    '''
        Using LeakyReLU in the encoder portion generates much 'cleaner'
        looking digits during MNIST PoC.
    '''
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.LeakyReLU(0.2)
    )

def dec_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.LeakyReLU(0.2)
    )

def conv_block(in_f, out_f):
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

def deconv_blocks(in_f, out_f):
    '''
        ConvTranspose blocks decrease the depth of the feature maps from
        in_f -> out_f. This sizing pattern is the opposite of the Conv_Blocks.
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(in_f, out_f, 2, stride = 2),
        nn.LeakyReLU(0.2)
    )
