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

from gan import Generator

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

class ewm_mlp(Generator):
    '''
        Per Chen et. al (2019), the MLP generator has the following
        architecture:
            Input(100) -> FC -> Hidden(512) -> FC -> Hidden(512)
                       -> FC -> Hidden(512) -> Output(D)
            Where D is the dimension of the output image.
        
        Here, EWM generator model is the same as the model used in the
        MLP GAN.
    '''
    pass