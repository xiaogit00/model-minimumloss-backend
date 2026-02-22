# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-v0_8')

PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
filename_without_ext = Path(__file__).stem

from utils.logging_config import setup_logger

logger = setup_logger(filename_without_ext)
# %%
training_data = datasets.FashionMNIST(
    root='../../data', # Downloads into folder called data in the same directory
    train=True, # Creates dataset from train-images-idx3-ubyte, otherwise from the t10k one. On first download, downloads both train and t10k (test) images
    download=True, # Will download if doesn't exist
    transform=ToTensor()
)
# %%

test_data = datasets.FashionMNIST(
    root='../../data', # Downloads into folder called data in the same directory
    train=False, # Creates dataset from train-images-idx3-ubyte, otherwise from the t10k one. On first download, downloads both train and t10k (test) images
    transform=ToTensor()
)

# %%
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
# %%
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
    
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
predictor = ShallowNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(predictor.parameters(), lr=1e-3)
train_loss_per_epoch = []
test_loss_per_epoch = []
train_error_percentage_per_epoch = []
test_error_percentage_per_epoch = []
#%%
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_correct = 0
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if batch == len(dataloader) - 1:
            print(f"batch: {batch}, appending train loss: {loss.item()}")
            train_loss_per_epoch.append(loss.item())
            train_correct /= len(dataloader.dataset)
            train_error_percentage_per_epoch.append(1 - train_correct)
            
# %%
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_loss_per_epoch.append(test_loss)
    test_error_percentage_per_epoch.append(1 - correct)

# %%
epochs = 20
for t in range(epochs):
    logger.info(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, predictor, loss_fn, optimizer)
    test(test_dataloader, predictor, loss_fn)
logger.info("Done!")
# %%
x = list(range(1, epochs+1))
plt.title("Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss values")
plt.plot(x, train_loss_per_epoch, label='Train Loss', marker='o', linestyle='--')
plt.plot(x, test_loss_per_epoch, label='Test Loss',marker='s',color="tomato", linestyle='--')
plt.legend()
plt.show()
# %%
x = list(range(1, epochs+1))
plt.title("Errors % per epoch")
plt.xlabel("Epoch")
plt.ylabel("Errors %")
plt.plot(x, train_error_percentage_per_epoch, label='Train Errors', marker='o', linestyle='--')
plt.plot(x, test_error_percentage_per_epoch, label='Test Errors',marker='s',color="tomato", linestyle='--')
plt.legend()
plt.show()