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
    
####
StockCNN(
  (layer1): Sequential(
    (0): Conv1d(5, 16, kernel_size=(5,), stride=(1,), padding=(1,))
    (1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): SiLU()
    (3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (layer2): Sequential(
    (0): Conv1d(16, 8, kernel_size=(5,), stride=(1,), padding=(1,))
    (1): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): SiLU()
    (3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc): Sequential(
    (0): Linear(in_features=104, out_features=20, bias=True)
    (1): SiLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=20, out_features=1, bias=True)
    (4): Sigmoid()
  )
)
