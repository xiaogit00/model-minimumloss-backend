# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import sys
from pathlib import Path

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
    root='../../data', # Downloads into folder called data in the parent same directory
    train=False, # Creates dataset from train-images-idx3-ubyte, otherwise from the t10k one. On first download, downloads both train and t10k (test) images
    transform=ToTensor()
)

# %%
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
# %%
class DeepNetwork(nn.Module):
    """
    A deep neural network with 3 hidden layers, of dimensions: 16, 12, 8
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # By default flattens everything from dim 1 onwards (retains batch size)
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 12)
        self.fc3 = nn.Linear(12, 8)
        self.fc4 = nn.Linear(8, 10)

    def forward(self, x):
        x = self.flatten(x) 
        f1 = self.fc1(x) 
        h1 = torch.relu(f1)
        f2 = self.fc2(h1) 
        h2 = torch.relu(f2)
        f3 = self.fc3(h2) 
        h3 = torch.relu(f3) 
        logits = self.fc4(h3) 
        return logits
    
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
predictor = DeepNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(predictor.parameters(), lr=1e-3)
#%%
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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

# %%
epochs = 80
for t in range(epochs):
    logger.info(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, predictor, loss_fn, optimizer)
    test(test_dataloader, predictor, loss_fn)
logger.info("Done!")
# %%