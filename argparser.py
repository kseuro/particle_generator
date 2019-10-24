############################################################################
# argparser.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.08.2019
# Purpose: - This python script is designed to provide model-agnostic command 
#            line argument parsing capabilities for PyTorch models associated
#            with the particle generator project.
#          - Functionality includes instantiation of argument parser objects
#            both for training and deploying PyTorch models.
############################################################################

# System Imports
from argparse import ArgumentParser

# Training argument parser function
def train_parser():
    '''
        Argument Parser Function for model training
        Does: prepares a command line argument parser object with
              functionality for training a generative model
        Args: None
        Returns: argument parser object
    '''
    usage = "Command line arguement parser for set up " + \
            "of PyTorch particle generator model training."
    parser = ArgumentParser(description=usage)
    
    # model: string that selects the type of model to be trained
    #        options: GAN, AE, EWM
    parser.add_argument('--model', type=str, default='GAN',
                        help='String that selects the model - options: \
                            GAN, AE, EWM | (default: &(default)s)')
    # checkpoint: string path to saved model checkpoint. If used with
    #             train function, model will resume training from checkpoint
    #             If used with deploy function, model will used saved weights.
    parser.add_argument('--checkpoint', type=str, default='',
                        help='String path to saved model checkpoint. If used \
                            with training function, model will resume trainig \
                                from that checkpoint. If used with deploy \
                                    function, model will deploy with save weights. \
                                        | (default: &(default)s) ')
    ################## Data Loading ######################
    ######################################################
    # data_root: path to training data folder (top level)
    parser.add_argument('--data_root', type=str, default='', 
                        help='Full path to training data folder \
                            | (default: &(default)s)')
    # save_root: path where training output is saved
    parser.add_argument('--save_root', type=str, default='/train_save',
                        help='Path where training output should be saved \
                            | (default: &(default)s)')
    # dataset: which LArCV1 dataset to use (512, 256, 128)
    parser.add_argument('--dataset', type=int, default=256,
                        help='Which crop size of the LArCV1 dataset to use \
                            | (default: &(default)s)')
    # batch_size: size of training data batches
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Size of data batches to use during model training\
                             | (default: &(default)s')
    # num_epochs: number of epochs to train model(s)
    parser.add_argument('--num_epoch', type=int, default=1,
                        help='Number of epochs over which to train the model(s)\
                             | (default: &(default)s)')
    # sample_batch: number of samples to generate during 
    #               training
    parser.add_argument('--sample_batch', type=int, default=8,
                        help='Number of image samples to be generated during\
                             training (progress check) | (default: &(default)s)')
    # gpu: which GPU to train the model(s) on
    parser.add_argument('--gpu', type=int, default=0,
                        help='Select gpu to use for training. If multi-gpu \
                            option is selected, then this option is ignored \
                                | (default: &(default)s)')
    # multi_gpu: toggle whether to train on multiple GPU's (if available)
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='Select whether to use multiple GPUs to train \
                            model. This model overrides the --gpu flag \
                                | (default: &(default)s)')
    # shuffle: toggle shuffle on/off
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Toggle dataloader batch shuffle on/off \
                            | (default: &(default)s)')
    # drop_last: toggle drop last batch on/off if dataset 
    #            size not divisible by batch size
    parser.add_argument('--drop_last', type=bool, default=False,
                        help='Toggle whether the dataloader should drop \
                            the last batch, if the dataset size is not \
                                divisible by the batch size \
                                    | (default: &(default)s)')
    # num_workers: number of worker threads for data io
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Set number of worker threads for data io \
                            | (default: &(default)s)')

    ################# Shared Settings ####################
    ######################################################
    # beta: beta value (for Adam optimizer only)
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Beta value for Adam optimizer \
                            | (default: &(default)s)')
    # momentum: momentum value (for SGD optimizer only)
    parser.add_argument('--p', type=float, default=0.9,
                        help='Momentum value for SGD optimizer \
                            | (default: &(default)s)')

    ################# Model settings #####################
    ######################################################
    ## Linear GAN Model
    # n_hidden: number of hidden units in each network
    parser.add_argument('--n_hidden', type=int, default=512,
                        help='Number of hidden units in each layer of G and D \
                            | (default: &(default)s)')
    # n_layers: number of layers of G and D
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of desired layers of G and D \
                            | (default: &(default)s)')

    ### Generator Network
    # g_lr: generator learning rate
    parser.add_argument('--g_lr', type=float, default=1e-4,
                        help='Generator network learning rate \
                            | (default: &(default)s)')
    # g_opt: generator optimizer
    parser.add_argument('--g_opt', type=str, default='adam',
                        help='Generator network optimizer function - \
                            choices: adam, sgd | (default: &(default)s)')
    # z_dim: dimension of G input vector
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Dimension of Generator network input vector \
                            size |(default: &(default)s)')

    ### Discriminator Network
    # d_lr: discriminator learning rate
    parser.add_argument('--d_lr', type=float, default=1e-4,
                        help='Discriminator network learning rate \
                            | (default: &(default)s)')
    # d_opt: discriminator optimizer
    parser.add_argument('--d_opt', type=str, default='adam',
                        help='Discriminator network optimizer function - \
                            choices: adam, sgd | (default: &(default)s)')

    ######################################################
    ## AE Model

    ### Encoder Network

    ### Decoder network


    ######################################################
    ## EWM Model
    # ewm_optim: optimizer function for EWM model training
    
    return parser

# Deploy model argument parser function
def deploy_parser():
    '''
        Argument Parser Function for model training
        Does: prepares a command line argument parser object with
              functionality for deploying a trained generative model.
        Args: None
        Returns: argument parser object
    '''
    usage = "Command line arguement parser for deploying " + \
            "trained PyTorch particle generator models."
    parser = ArgumentParser(description=usage)

    # TODO: Write the parser arguments after deciding on a convention for how 
    # model outputs, checkpoints, and experiments will be saved.

    return parser
