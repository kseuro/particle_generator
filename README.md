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
Our working hypothesis is that the non-zero pixels in LArCV1 monte carlo data images is define only on a very thin hyper-manifold extending through the image data space. That is, the particle charge information lies on a thin manifold in NxN space. Most of the pixels within this space have a value very close to zero and results in convolutional implementations of GANs to collapse to modes that produce only black images. In the case of a GAN, there appears to be difficult to find a direct mapping from the space of the Generator's 100-dimensional Gaussian input noise to the space of NxN LArCV1 data images, using conventional GAN training methods.

Instead, we propose to use an AutoEncoder to learn a compact representation of the LArCV1 data, and subsequently use the model's learned latent representation as a target for training a generative model. The generative model will then be trained to reproduce images in data space using the non-adversarial EWM approach.

### Short-term goal(s):
- Train for single iteration using MNIST and save outputs (Done)
- GAN proof-of-concept using MNIST (Done)
- AE proof-of-concept using MNIST (Done)
- EWM proof-of-concept using MNIST (Done - See EWM repo)

### Repo work in progress:
- model agnostic argparser (Done)
- data loading class - batch-to-batch (Done)
- data loading class - EWM single batch (Done)
- data loading function (Done)
- Training routine (Done)
    - LArCV data training funcitons
        - GAN (Done)
        - AE (Done)
    - Code-vector targets training functionality
        - EWM
- Sample saving functionality
  - GAN (Done)
  - AE (Done)
- Metrics saving functionality (Done)
- Checkpoint saving functionality (Done)

#### Deploy functionality
- Deploy routines
    - GAN
    - AE (Done)
    - EWM

#### GAN
- linear GAN model class (Done)
- linear GAN training function (Done)

#### EWM
- move model and functionality from EWM repo (Done)
- Load code vector targets as torch tensors

#### AE
- AE model class (Done)
- AE training function (Done)
- AE deploy function (Done)

#### Model evaluation functionality
- graphics generation capabilities
    - Loss Plotting (Done)
- hamming distance (on GPU) computations
    - Move functionality over from EWM repo

### Requirements:
- Python version 3.5 or later
- PyTorch version 1.0 or later
- CUDA version 10.0 or later (no CPU implemention is provided here)

### Experiment Requirements
The ArgParser requirements for each respective experiment are listed below:
- All Models:
    - Dataloader kwargs: <pre><code>--data_root --save_root --dataset --batch_size --num_epochs --sample_size --gpu --shuffle --drop_last --num_workers </code></pre>
    - Model and optim specs: <pre><code>  --n_layers --beta --p </code></pre>
- GAN Model:
    - <pre><code> --n_hidden --g_lr --g_opt --z_dim --d_lr --d_opt </code></pre>
- AutoEncoder Model:
    - <pre><code> --l_dim --ae_opt --ae_lr </code></pre>
- EWM Training Routing
    - <pre><code> --g_lr --g_opt --n_hidden --psi_lr --mem_size </code></pre>


