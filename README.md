# Particle Generator
Deep generative models applied to 2D particle interaction image generation.

This repository contains PyTorch implementations of generative models applied to Liquid Argon Computer Vision 1 (LArCV1) datasets.

### Generative-Adversarial Network (GAN)
The GAN is two separate multi-layer perceptron models. This model implementation will provide a base-line for hypothesis testing, iterative development, benchmarking of model performance, and overall proof-of-concept (PoC) demonstrations.

### Explicit Wasserstein Minimization (EWM)
This model consists of a single generator function to be trained using explicit wasserstein minimization as described in:

@article{1906.03471, Author = {Yucheng Chen and Matus Telgarsky and Chao Zhang and Bolton Bailey and Daniel Hsu and Jian Peng}, Title = {A gradual, semi-discrete approach to generative network training via explicit Wasserstein minimization}, Year = {2019}, Eprint = {arXiv:1906.03471},

### Auto-Encoder (AE)
Our working hypothesis is that the non-zero pixels in LArCV1 monte carlo data images is define only on a very thin hyper-manifold extending through the image data space. That is, the particle charge information lies on a thin manifold in NxN space. Most of the pixels within this space have a value very close to zero and results in convolutional implementations of GANs to collapse to modes that produce only black images. In the case of a GAN, it appears to be difficult to find a direct mapping from the space of the Generator's 100-dimensional Gaussian input noise to the space of NxN LArCV1 data images using conventional GAN training methods.

Instead, we propose to use an AutoEncoder to learn a compact representation of the LArCV1 data, and subsequently use the model's learned latent representation as a target for training a generative model. We employ both Multilayer-Perceptron and Fully-Convolutional implementations of the AutoEncoder, in order to determine which class of model provides the best image resolution and generalizable reconstruction.

### Experimental Summary

Once an AutoEncoder is fully trained, we can generate a set of code vectors from the bottlebneck layer of the model, thereby creating a target dataset that we want to model using a Generator function. A generative model will then be trained to reproduce the AutoEncoder's set of code vectors using non-adversarial training. A Generator function trained in this way can then be used as a generalized input to the Decoder branch of a trained AutoEncoder model in order to synthesize novel particle decay event images.

### Workflow

In order to produce trained AutoEncoder and Generative models capable of functioning together to produce novel particle interaction images, we use the following workflow:

1. Train an AutoEncoder
    - Select a LArCV image dataset for which you wish to learn a compressed representation.
        - Arg conversion example: <pre><code> --dataset 256 </code></pre> = Set of 256x256 LArCV1 training images
    - Select either an mlp or convoutional implementation by using <pre><code> --model ae </code></pre> or <pre><code> --model conv_ae </code></pre> respectively in the argument list in the model training script.

2. Evaluate the Trained AutoEncoder Model
    - Evaluate the MeanSquaredError loss for checkpoints: {600, 650, 700, 750, 800, 850, 900, 999} using both the test dataset and the reference training dataset.
        - Note: The reference training dataset contains the same number of images as the test dataset, in order to put the loss calculations on the same footing with eachother. 
        - For each checkpoint, evaluate the difference in loss between the reference training examples and the test examples.
        - The Decoder state we wish to use is the one which corresponds to the smallest difference in MSE before the train and test curves begin to diverge.
        - From the checkpoint selected in the above process, generate a set of model outputs using the test dataset, in order to visually evaluate the image reconstruction quality for that checkpoint.
    - Generate a set of code vector targets using the checkpoint selected above, one vector for each training example in the full training dataset.
        - If using an MLP AutoEncoder, you can simply take the output of the model at the bottleneck layer and store the result as a NumPy array.
        - If using a Conv AutoEncoder, the output of the bottleneck layer needs to be flattened into vector before being stored as a NumPy array.
    
3. Train a generative model using the EWM algorithm
    - Select a set of code vectors to which you want to fit a Generator function and train.

4. Couple the trained Generator model to the Decoder branch of a trained AutoEncoder
    - The trained Generator model can now be used as input to a trained Decoder model in order to generate images.

### Experiment Requirements
The ArgParser requirements for each respective experiment are listed below:
- All Models:
    - Dataloader kwargs: <pre><code>--data_root --save_root --dataset --batch_size --num_epochs --sample_size --gpu --shuffle --drop_last --num_workers </code></pre>
    - Model and optim specs: <pre><code>  --n_layers --beta --p </code></pre>
- GAN Model:
    - <pre><code> --n_hidden --g_lr --g_opt --z_dim --d_lr --d_opt </code></pre>
- AutoEncoder Model:
    - Both MLP and Convolutional:
        - <pre><code> --l_dim --ae_opt --ae_lr </code></pre>
    - Additional args for convolutional model:
        - <pre><code> --depth </code></pre>
- EWM Training Routing
    - For training on the outputs of both MLP and Convolutional AutoEncoder:
        - <pre><code> --g_lr --g_opt --n_hidden --psi_lr --l_dim --mem_size --vec_root </code></pre>
        - **Note:** <pre><code>--vec_root </code></pre> specifies the location of a set of code vectors produced by the AutoEncoder using the TRAINING data. This set of vectors is used only to compute the stopping criterion in the EWM training routine.
    - Additional arguments for training on the output of a Convolutional AutoEncoder:
        - <pre><code> --ewm_target </code></pre>

### Software and Hardware Requirements:
- Python version 3.5 or later
- PyTorch version 1.0 or later
- CUDA version 9.2 or later
- One CUDA enabled GPU with **at least** 3GB of memory
