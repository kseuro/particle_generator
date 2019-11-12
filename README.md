# Particle Generator
Deep generative models applied to 2D particle interaction image generation.

This repository contains PyTorch implementations of generative models applied to liquid Argon computer vision 1 (LArCV1) datasets.

**Models and scripts under construction at this time:**

### Generative-Adversarial Network (GAN)
GAN as two separate multi-layer perceptron models. This model implementation will provide a base-line for hypothesis testing, iterative development, benchmarking of model performance, and overall proof-of-concept (PoC) demonstrations. PoC will be carried out using the MNIST dataset of handwritten digits, as this dataset most closely mimmicks the LArCV1 dataset. 

### Explicit Wasserstein Minimization (EWM)
This model consists of a single generator function to be trained using explicit wasserstein minimization as described in:

@article{1906.03471, Author = {Yucheng Chen and Matus Telgarsky and Chao Zhang and Bolton Bailey and Daniel Hsu and Jian Peng}, Title = {A gradual, semi-discrete approach to generative network training via explicit Wasserstein minimization}, Year = {2019}, Eprint = {arXiv:1906.03471},


### Auto-Encoder (AE)
Our working hypothesis is that the non-zero pixels in LArCV1 monte carlo data images is defined only on a very thin hypermanifold extending through the high dimensional (e.g. (512x512)) data space. Most of the pixels within this space have a value very close to zero, which causes convolutional implementations of GANs to converge towards generative modes that produce only black images. In other words, there appears to be no simple, direct mapping from the space of the Generator's 100-dimensional Gaussian input noise to the space of NxN LArCV1 data images, using only a GAN approach. (Comments and criticism welcome).

Instead, we propose to use a AE to learn a compact representation of the LArCV1 data distribution, and subsequently use the model's learned latent as input to a generative model. The generative model will then be trained to reproduce images in data space using a non-adversarial approach.

### Short-term goal(s):
- Train for single iteration using MNIST and save outputs
- Resume training of a saved model for a single iteration 
- GAN proof-of-concept using MNIST
- AE proof-of-concept using MNIST
- EWM proof-of-concept using MNIST

### Repo work in progress:
- model agnostic argparser (Done)
- data loading class - batch-to-batch (Done)
- data loading class - EWM single batch
- data loading function (Done)
- training routine
- model specific training functions
  - GAN
  - EWM
  - AE
- sampling, output, and saving

#### Proof-of-Concept
- training script for running linear GAN model on MNIST dataset
- training script for running VAE on MNIST dataset

#### Launch functionality
- Deploy scripts
- Evaluation scripts

#### GAN
- linear GAN model class (Done)
- linear GAN training function (Done)
- linear GAN deploy function

#### EWM
- move model and functionality from EWM repo

#### AE
- AE model class
- AE training function
- AE deploy function

#### Model evaluation functionality
- evaluation of metrics from saved training states
- graphics generation capabilities
- hamming distance (on GPU) computations

### Requirements:
- Python version 3.5 or later
- PyTorch version 1.0 or later
- CUDA version 10.0 or later (no CPU implemention will be provided here)
