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
from gan import Print

class ewm_mlp(Generator):
    '''
        Per Chen et. al (2019), the MLP generator has the following
        architecture:
            Input(100) -> FC -> Hidden(512) -> FC -> Hidden(512)
                       -> FC -> Hidden(512) -> Output(D)
            Where D is the dimension of the output image.

        Here, EWM generator model is the same as the model used in the
        MLP GAN.
        - This class simply inherits the Generator class from gan.py
    '''
    pass
