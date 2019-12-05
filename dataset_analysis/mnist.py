###############################################################################
# MNIST.py
# Author: Kai Kharpertian
# Organization: Tufts University
# Department: Physics
# Date: 12.05.2019
# Purpose: This script takes the MNIST dataset and computes the percentage of
#          each image that contains a value greater than zero.
#          The percentage is plotted a function of the sum of the binarized
#          vector representation of that same image.
###############################################################################
import torch
from torchvision      import datasets
from torchvision      import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
# Import the MNIST dataset
config = {'num_workers': 8, 'batch_size': 1, 'shuffle': False, 'drop_last': False}

transform = transforms.Compose( [transforms.ToTensor(),
                                 transforms.Normalize([0.5],[0.5])] )
data = datasets.MNIST(root='./data', train=True, download=True,
                      transform=transform)
dataloader = tqdm(DataLoader(data, **config))

results = []
# Loop over the dataset
for itr, (data, _) in enumerate(dataloader):
    # Flatten the image into a vector
    data = data.view(-1)

    # Binarize the image
    min = torch.tensor([0]); max = torch.tensor([1])
    data = torch.where(data > 0.0, max, min)

    # Get the sum and fill percentage
    sum = data.sum()
    p_fill = sum / data.size(0)

    # Append result
    results.append( (sum, p_fill) )

    # Testing
    if itr % 10 == 0:
        print(results)
        break
