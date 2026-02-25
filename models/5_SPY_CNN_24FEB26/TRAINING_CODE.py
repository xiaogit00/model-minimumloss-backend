# %% ############## Imports and definitions ################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import os, sys
from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
filename_without_ext = Path(__file__).stem
from utils.logging_config import setup_logger

logger = setup_logger(filename_without_ext)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=20):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Input: Sequence of 'seq_len' days
        x = torch.tensor(self.X[idx : idx + self.seq_len], dtype=torch.float32).transpose(0, 1)
        # Target: The target at the end of the sequence
        y = torch.tensor(self.y[idx + self.seq_len], dtype=torch.float32)
        return x, y

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to {seed}")

def plot_results(train_loss, test_loss, train_acc, test_acc, title="Model Results"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(train_loss, label='Train Loss', color='blue')
    axs[0].plot(test_loss, label='Test Loss', color='orange')
    axs[0].set_title(f'{title} - Loss')
    axs[0].legend()
    axs[1].plot(train_acc, label='Train Acc', color='green')
    axs[1].plot(test_acc, label='Test Acc', color='red')
    axs[1].set_title(f'{title} - Accuracy')
    axs[1].legend()
    plt.show()


def train_model_with_scheduler(model, lr, train_loader, test_loader, num_epochs=50):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in tqdm(range(num_epochs)):
        if epoch == 0:
            logger.info("MODEL ARCHITECTURE")
            logger.info(model)
            logger.info('\n-------------------\n')
            logger.info("INITIAL MODEL WEIGHTS")
            logger.info(f"First Conv1D Weights: {model.layer1[0].weight.shape}:\n {model.layer1[0].weight}")
            logger.info('\n-------------------\n')
            logger.info(f"Second Conv1D Weights: {model.layer2[0].weight.shape}:\n {model.layer2[0].weight}")
            logger.info('\n-------------------\n')
            logger.info(f"Hidden Layer 3 (Linear) Weights: {model.fc[0].weight.shape}:\n {model.fc[0].weight}")
            logger.info('\n-------------------\n')
            logger.info(f"Output layer Weights: {model.fc[3].weight.shape}:\n {model.fc[3].weight}")
            logger.info('\n-------------------\n')
        
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            y = y.view_as(out)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            preds = (out > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_losses.append(run_loss/len(train_loader))
        train_accs.append(correct/total)
        scheduler.step()
        if epoch == num_epochs-1:
            logger.info("FINAL MODEL WEIGHTS")
            logger.info(f"First Conv1D Weights: {model.layer1[0].weight.shape}:\n {model.layer1[0].weight}")
            logger.info('\n-------------------\n')
            logger.info(f"Second Conv1D Weights: {model.layer2[0].weight.shape}:\n {model.layer2[0].weight}")
            logger.info('\n-------------------\n')
            logger.info(f"Hidden Layer 3 (Linear) Weights: {model.fc[0].weight.shape}:\n {model.fc[0].weight}")
            logger.info('\n-------------------\n')
            logger.info(f"Output layer Weights: {model.fc[3].weight.shape}:\n {model.fc[3].weight}")
            logger.info('\n-------------------\n')

        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                y = y.view_as(out)
                loss = criterion(out, y)
                test_loss += loss.item()
                preds = (out > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
        test_losses.append(test_loss/len(test_loader))
        test_accs.append(correct/total)

    return train_losses, test_losses, train_accs, test_accs

# %% ############## MODEL DEFINITION ################

# Complete the StockCNN class by filling in the blanks; (Do Not Modify Others)
class StockCNN(nn.Module):
    def __init__(self, input_dim=40, kernel_size=3):
        super(StockCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=kernel_size, padding=1),  # < Complete the code by filling in the correct parameter >
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(8),
            nn.SiLU(),
            nn.MaxPool1d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(20, 1),  # < Complete the code by filling in the correct parameter >
            nn.Sigmoid()
        )
    def forward(self, x):
        # print("x size:", x.size())
        out = self.layer1(x)
        # print("layer1 out size:", out.size())
        out = self.layer2(out)
        # print("layer2 out size:", out.size())
        out = self.flatten(out)
        # print("flatten out size:", out.size())
        return self.fc(out)

# %% ############## DATASET PREPARATION ################

data = pd.read_csv('../../finance_timeseries/SPY_15y.csv', header=0, skiprows=[1,2], index_col=0, parse_dates=True)

features = data[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change()
target = ((data['Close'] - data['Close'].shift()) > 0).astype(int)

# Drop NAs created by pct_change (Do Not Modify)
combined = pd.concat([features, target], axis=1).dropna()
features_change = combined.iloc[:, :-1].values
target_change = combined.iloc[:, -1].values
seq_len_long = 60
# Scaling
# < Complete the code here >
scaler = StandardScaler()
features_change = scaler.fit_transform(features_change)

train_size = int(len(features_change) * 0.8)

# Re-instantiate datasets with new sequence length
train_dataset_long_change = TimeSeriesDataset(features_change[:train_size], target[:train_size], seq_len_long)
test_dataset_long_change = TimeSeriesDataset(features_change[train_size:], target[train_size:], seq_len_long)

# Create Loaders
set_seed(42)
train_loader_long_change = DataLoader(train_dataset_long_change, batch_size=32, shuffle=True)
test_loader_long_change = DataLoader(test_dataset_long_change, batch_size=32, shuffle=False)

# Train the model and plot the results
def L(L_in, K):
    return (L_in + 2 - (K-1) - 1)+1
set_seed(42)
k = 5
L_out = L((L(60, k))//2, k)//2

# %% ############## MODEL TRAINING ################
model_sched = StockCNN(8*L_out, k)

#%%
t_loss, v_loss, t_acc, v_acc = train_model_with_scheduler(model_sched, 1e-3, train_loader_long_change, test_loader_long_change)
plot_results(t_loss, v_loss, t_acc, v_acc, title=f"Raw Prices for {k} kernel size")
# %%
