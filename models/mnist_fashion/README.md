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

---
### Model

```python
class ShallowNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # By default flattens everything from dim 1 onwards (retains batch size)
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.flatten(x) # (64, 28, 28) -> (64, 784)
        Z1 = self.fc1(x) # (64, 784) -> (64, 16)
        A1 = torch.relu(Z1) #(64, 16)
        logits = self.fc2(A1) #(64, 10)
        return logits #(64, 10)
```

### Optimizer 

`optimizer = torch.optim.SGD(predictor.parameters(), lr=1e-3)`

### Loss Function

`loss_fn = nn.CrossEntropyLoss()`

```python
pred = model(X) # (64, 10) -> logits
loss = loss_fn(pred, y) # (64, 10) -> class probabilities 
```

It's important to note that `nn.CrossEntropyLoss()` *always*` expects raw logits for your `pred` argument, and `y` to be class indices. 

Internally, it does softmax on the raw logits, and calculates NLLLoss. The result is a scalar. As a rule of thumb, if you're using nn.CrossEntropyLoss, NEVER apply softmax in your model. The final layer of your model should end with `self.fc = nn.Linear(...)` and not `nn.Softmax(dim=1)`. 

Here's how you can inspect whether something is logits or class probabilities:

```python
>>> logits[0]

tensor([ 0.1313,  0.2775,  0.1928, -0.4062,  0.0720, -0.4089, -0.2134,  0.0554,
        -0.1037, -0.1687], grad_fn=<SelectBackward0>)
```

Notice that negative numbers exist for the logits, and they don't sum up to 1. 

If you do a softmax over the logits, which is what the loss function does implicitly,

`preds = softmax(logits)`

...You'll see that the output will be positive and they'll sum up to one: 

```python
>>> preds = softmax(logits)
>>> preds[0]
tensor([0.1177, 0.1362, 0.1252, 0.0688, 0.1109, 0.0686, 0.0834, 0.1091, 0.0930,
        0.0872], grad_fn=<SelectBackward0>)

>>> preds[0].sum()
tensor(1., grad_fn=<SumBackward0>)
```








