### Dataset

Download using `torchvision.datasets.FashionMNIST()`

```python
training_data = datasets.FashionMNIST(
    root='data', # Downloads into folder called data in the same directory
    train=True, # Creates dataset from train-images-idx3-ubyte, otherwise from the t10k one. On first download, downloads both train and t10k (test) images
    download=True, # Will download if doesn't exist
    transform=ToTensor()
)
```
Data folder containing: 
```
training_code/
├── data/
│   ├── FashionMNIST/raw
│       └── t10k-images-idx3-ubyte  
│       └── t10k-labels-1dx1-ubyte
│       └── train-images-idx3-ubyte
│       └── train-labels-idx1-ubyte
```
idx3 = 3D array; idx1 = 1D array

If you remove the .gz files, basically, it's train image and labels, test image and labels. 


#### Training Dataset
The data extends the base Dataset Object:
```python
>>> training_data.__class__.__mro__

(torchvision.datasets.mnist.FashionMNIST,
 torchvision.datasets.mnist.MNIST,
 torchvision.datasets.vision.VisionDataset,
 torch.utils.data.dataset.Dataset,
 typing.Generic,
 object)

>>> training_data

> Dataset FashionMNIST
    Number of datapoints: 60000
    Root location: data
    Split: Train
    StandardTransform
Transform: ToTensor()
```

#### Test Dataset
Has the same MRO as train_data. 

```python
>>>test_data

Dataset FashionMNIST
    Number of datapoints: 10000
    Root location: data
    Split: Test
    StandardTransform
Transform: ToTensor()
```

If you don't apply the `ToTensor()` transforms, the output will be a PIL image:

```python
>>> test_data.__getitem__(1)

(<PIL.Image.Image image mode=L size=28x28>, 2)
```

#### Inspecting one datapoint:
```python
>>> img, label = training_data[0]
>>> img.shape
torch.Size([1, 28, 28])
>>> label
9
``` 

#### Getting class names:
```python
>>> training_data.classes
['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']
```


#### Sampling multiple datapoints:
```python
import matplotlib.pyplot as plt

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1): # generate 9 images
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # This literally just generates a random integer within 60k training_data and returns the scalar
    img, label = training_data[sample_idx] #gets X and y (y is scalar)
    figure.add_subplot(rows, cols, i) # adds image to index i of subplots
    plt.title(labels_map[label]) # use the labels as title
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray") #Imshow adds the data as image. the data is tensor of shape (1, 28, 28). squeeze makes it shape (28, 28)
plt.show()
```
#### DataLoaders

We'll load it into DataLoaders with batch size 64:

```python
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```

Where later on, we'll do mini-batch SGD like so:

```python
for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        ...
```