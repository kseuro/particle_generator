###############################################################################
# dataloader.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 10.15.2019
# Purpose: - This file provides dataloading functionality for training
#            a generative model using the LArCV1 dataset
#          - The LArCV1 dataloader class inherits from the torch absract
#            dataset class: torch.utils.data.Dataset
#               - The following methods are overriden:
#                   - __len__ so that len(dataset) returns the dataset size
#                   - __getitem__ to support indexing into dataset, e.g. so
#                                 that dataset[i] gets the ith data sample.
#          - The BottleLoader class inherits from the torch abstract
#            dataset class: torch.utils.Dataset
#               - This class allows the code-vector targets generated from
#                 the bottleneck layer of a trained AutoEncoder to be used
#                 as training data for a Generator model.
#               - The follow packages need to be installed for the BottleLoader
#                 class to work: pandas (for csv parsing)
#               - The following methods are overriden:
#                   - __len__ so that len(dataset) returns the dataset size
#                   - __getitem__ to support indexing into dataset, e.g. so
#                                 that dataset[i] gets the ith data sample.
###############################################################################

# Imports
import os
import PIL
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# string tuples for exception handling
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
                  '.tif', '.tiff', '.webp')
VALID_DSETS = ('larcv_png_512', 'larcv_png_256', 'larcv_png_128',
               'larcv_png_64' , 'larcv_png_32', 'code_vectors')
CONV_FLAGS = ('RGB', 'L')

# Dataloader constructor functions
def verify_image(image_path):
    '''
        Does: verifies that a training image is an image (e.g. not corrupt)
        Args: image_path (string): full path to training image
        Returns: boolean: true if image, false otherwise
    '''
    try:
        img = Image.open(image_path)  # open the image file
        img.verify()  # verify that it is, in fact an image
        return True
    except (IOError, SyntaxError) as e:
        print('Bad file:', image_path)
        return False

def dset_tag(root):
    '''
        This function assumes that the training dataset is one of the following:
            - [larcv_png_512, larcv_png_256, larcv_png_128, larcv_png_64]
        The folder structure for each dataset is assumed to be:
            - larcv_png_xxx/larcv_png_xxx/<all images>
        All training images are assumed to be located in the second data folder,
            since all of the training images are of the same class.
        Does: adds the approriate dataset tag to the data root.
        Args: root (string): full path to the selected LArCV1 dataset
        Returns: root (string) with appropriate dataset tag
    '''
    # Tag for code_vector target data (EWM generator model targets)
    if 'code' in root:
        return root + 'code_vectors/'
    # Tag for LArCV1 image data (AutoEncoder or GAN model targets)
    if str(512) in root:
        return root + 'larcv_png_512/'
    elif str(256) in root:
        return root + 'larcv_png_256/'
    elif str(128) in root:
        return root + 'larcv_png_128/'
    elif str(64) in root:
        return root + 'larcv_png_64/'
    elif str(32) in root:
        return root + 'larcv_png_32/'
    else:
        raise(RuntimeError('Invalid dataset selection. Valid datasets are:'
                            + ','.join(VALID_DSETS)))

def get_paths(root):
    '''
        Does: gets the full path for every training example in a dataset
        Args: root (string): full path to folder of training examples
        Returns: list of full paths (strings) to training examples
    '''
    # Get appropriate dataset tag
    root = dset_tag(root)
    paths = []
    larcv = True if 'larcv' in root else False

    # Walk through image folder and compute paths
    for example in os.listdir(root):
        example_path = os.path.join(root, example)
        if larcv:
            if verify_image(example_path):
                paths.append(example_path)
            else:
                continue
        else:
            paths.append(example_path)

    # Make sure dataloading occured
    if larcv:
        if len(paths) == 0:
            raise(RuntimeError("Found 0 LArCV Images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    elif len(paths) == 0:
            raise(RuntimeError("Found 0 code_vector files in subfolders of: {}".format(root)))

    # Debugging Statements
    print("Testing dataloader.get_paths() function")
    print("Length of paths array: ", len(paths))
    print("Sample of the first 10 paths in the array: ")
    for i in range(10):
        print(paths[i])
    input(...)
    return paths

# Dataloading functions
def pil_loader(image_path, conv_flag):
    '''
        Does: loads an image as a pillow file and converts the image to color
              if 'RGB' conversion flag is selected, or gray scale if 'L' flag
              is selected
        Args: - image_path (string): full path to data image
              - conv_flag (string): image conversion selection. See above.
        Returns: converted data image
    '''
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        return img.convert(conv_flag)

# Dataset Class - batch-to-batch loading of LArCV images
class LArCV_loader(Dataset):
    '''
        Liquid Argon Computer Vision dataloader class
        Does: Creates a dataloader object that inherits from the base
              PyTorch nn.Dataset class, for loading LArCV1 dataset images.
        Args: - root (string): full path to the training images
              - transform (callable, optional): function that takes a PIL image
                    and returns a transformed version.
        Returns: LArCV1 dataloader object equipped with image transforms.

        The dataloader object expects the following image-folder structure:
            full_path/image_class/image0.png
            full_path/image_class/image1.png
            .
            .
            .
            full_path/image_class/imageN.png
        That is, all images in the LArCV1 dataset are of the same class,
        e.g. particle interaction.
    '''

    def __init__(self, root, transforms=None, conv_flag='L'):
        self.root       = root
        self.paths      = get_paths(self.root)
        self.transforms = transforms
        self.conv_flag  = conv_flag
        if self.conv_flag not in CONV_FLAGS:
            raise(RuntimeError("Conversion flag not recognized. " + "\n"
                "Valid conversion flags are:" + ",".join(CONV_FLAGS)))
        print('Image conversion flag is: {}'.format(self.conv_flag))
        print('Images will be loaded from subfolder of: {}'.format(self.root))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        '''
            Does: loads a data image into dataloader object
            Args: index (int): Index of data image
            Returns: - data image loaded using pil_loader function
                     - transformed data image if transform is not None
        '''
        image = pil_loader(self.paths[index], self.conv_flag)

        if (self.transforms is not None):
            image = self.transforms(image)

        return image

# Dataset class - batch-to-batch loading of AE.Encoder code-vectors
class BottleLoader(Dataset):
    '''
        BottleLoader class for handling the loading of code-vector
        targets produced by the Encoder branch of a trained AutoEncoder
        model.
        Does: Creates a dataloader object that inherits from the base
              PyTorch nn.Dataset class, for loading target .csv files
              as Torch Tensors.
        Args: - root (string): full path to the code-vector .csv files
              - transform (callable): optional transform to be called on a
                                      code vector training example.
        The dataloader object expects the following image-folder structure:
            full_path/code_vectors_{dataset}_{l_dim}/code_vectors_{dataset}_{l_dim}/target_0.csv
            full_path/code_vectors_{dataset}_{l_dim}/code_vectors_{dataset}_{l_dim}/target_1.csv
            .
            .
            .
            full_path/code_vectors_{dataset}_{l_dim}/code_vectors_{dataset}_{l_dim}/target_N.csv
    '''
    def __init__(self, root, transforms=None):
        self.root = root
        self.csv_paths = get_paths(self.root)
        self.transforms = transforms
        print("Code-Target examples will be loaded from subfolder of: {}".format(self.root))

    def __len__(self):
        return len(self.csv_paths)

    def __getitem__(self, index):
        code_vector = np.genfromtxt(self.csv_paths[index], delimiter=',')

        if self.transform is not None:
            code_vector = self.transforms(code_vector)

        return code_vector
