# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# %%
training_data = datasets.FashionMNIST(
    root='data', # Downloads into folder called data in the same directory
    train=True, # Creates dataset from train-images-idx3-ubyte, otherwise from the t10k one. On first download, downloads both train and t10k (test) images
    download=True, # Will download if doesn't exist
    transform=ToTensor()
)
# %%

test_data = datasets.FashionMNIST(
    root='data', # Downloads into folder called data in the same directory
    train=True, # Creates dataset from train-images-idx3-ubyte, otherwise from the t10k one. On first download, downloads both train and t10k (test) images
    transform=ToTensor()
)

# %%
training_data.__class__.__mro__
# %%
training_data