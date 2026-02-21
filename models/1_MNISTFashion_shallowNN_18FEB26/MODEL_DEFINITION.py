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