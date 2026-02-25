# %%
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
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

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
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in tqdm(range(num_epochs)):
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
        # scheduler.step()

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

# %%

data = pd.read_csv('../../data/finance_timeseries/SPY_15y.csv', header=0, skiprows=[1,2], index_col=0, parse_dates=True)

# %%
# --- EXPERIMENT B: RETURN DATA ---

features = data[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change()
target = ((data['Close'] - data['Close'].shift()) > 0).astype(int)

# Drop NAs created by pct_change (Do Not Modify)
combined = pd.concat([features, target], axis=1).dropna()
features_change = combined.iloc[:, :-1].values
target_change = combined.iloc[:, -1].values

# Scaling
# < Complete the code here >
scaler = StandardScaler()
features_change = scaler.fit_transform(features_change)

# Split 80/20
# < Complete the code here >
train_size = int(len(features_change) * 0.8)
train_dataset = TimeSeriesDataset(features_change[:train_size], target_change[:train_size])
test_dataset = TimeSeriesDataset(features_change[train_size:], target_change[train_size:])

# Create Dataloader
set_seed(42) # Reset seed for fair comparison
# < Complete the code here >
train_loader_raw = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader_raw = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train the model and plot the results
set_seed(42) # Reset seed for fair comparison
# < Complete the code here >
model_raw = StockCNN()
t_loss, v_loss, t_acc, v_acc = train_model_with_scheduler(model_raw, 1e-3, train_loader_raw, test_loader_raw)
plot_results(t_loss, v_loss, t_acc, v_acc, title="Raw Prices")
# %%
