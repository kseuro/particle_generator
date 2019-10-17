###############################################################################
# train.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.15.2019
# Purpose: - This file provides a model agnostic training routine. Training is
#            carried out as standard batch-to-batch training for both the
#            GAN and VAE. Each model's respective training function is loaded
#            at runtime from train_fns.py
###############################################################################

# Imports

# Load state dict (optional)

# Create dataloader parameter dictionary
# Create dataloader

# Create optimizer parameter dictionary
# Update config with D in_features based on image size

# Instantiate desired model (GAN or VAE)
# Perform weight initialization
# G_D should be None type if not parallel

# Update config based on model selection

# Generate key word arguments dict for training function
# G, D, G_D, z_fixed, loss_fn

# Setup training function (pass config as well)

# Train model for specified number of epochs

