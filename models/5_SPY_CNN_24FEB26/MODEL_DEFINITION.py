class StockCNN(nn.Module):
    def __init__(self, input_dim=40):
        super(StockCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=3, padding=1),  # < Complete the code by filling in the correct parameter >
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
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
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        return self.fc(out)