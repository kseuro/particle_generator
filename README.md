# Particle Generator
Deep generative models applied to 2D particle interaction image generation.

This repository contains PyTorch implementations of generative models applied to liquid Argon computer vision 1 (LArCV1) datasets.

**Models under construction at this time:**

### Generative-Adversarial Network (GAN)
GAN as two separate multi-layer perceptron models. The model implementation will provide a base-line for hypothesis testing, itertive development, benchmarking of model performance, and overall proof-of-concept (PoC). PoC will be carried out using the MNIST dataset of handwritten digits, as this dataset most closely mimmicks the LArCV1 dataset. 

### Variational Auto-Encoder (VAE)
Our working hypothesis is that the non-zero pixels in LArCV1 monte carlo data images is defined only on a very thin hypermanifold extending through the high dimensional (e.g. [512x512]) data space. Most of the pixels within this space have a value very close to zero, which causes convolutional implementations of GANs to converge towards generative modes that produce only black images. In other words, there is no easy, direct mapping that can be learned from the space of 100-dimensional Gaussian input noise to the space of nxn LArCV1 data images using only a GAN approach.

Instead, we propose to use a VAE to learn a compact representation of the LArCV1 data, and use the latent space learned by the model as inputs to a generative model. The generative model will then be trained to reproduce images in data space using a non-adversarial approache.


### Requirements:
- Python version 3.5 or later
- PyTorch version 1.0 or later
- CUDA version 10.0 or later (no CPU implemention will be provided here)
