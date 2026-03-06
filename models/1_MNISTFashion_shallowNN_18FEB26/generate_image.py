# %%
from torchvision import datasets
from torchvision.transforms import ToTensor


test_data = datasets.FashionMNIST(
    root='../../data', # Downloads into folder called data in the same directory
    train=False, # Creates dataset from train-images-idx3-ubyte, otherwise from the t10k one. On first download, downloads both train and t10k (test) images
    download=True, # Will download if doesn't exist
)
# %%
for i in range(1, 41):
    test_data.__getitem__(i)[0].save(f'./images/mnist_{i}.jpg')

# %%
