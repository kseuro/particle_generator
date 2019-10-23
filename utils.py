###############################################################################
# utils.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.16.2019
# Purpose: - This file provides setup functionality for model training,
#            resuming training from checkpoint, metric tracking, and
#            saving outputs.
###############################################################################

# Imports
import os

# Model agnostic functionality
def make_dir(dir):
    '''
        Simple utility function for creating directories
    '''
    if os.path.exists(dir):
        return
    try:
        os.makedirs(dir)
    except OSError as E:
        if E.errno != errno.EEXIST:
            raise



# GAN Functionality

# EWM Functionality

# AE Functionality
