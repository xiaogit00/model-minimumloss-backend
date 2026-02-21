class DeepNetwork(nn.Module):
    """
    A deep neural network with 3 hidden layers, of dimensions: 8, 8, 8
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # By default flattens everything from dim 1 onwards (retains batch size)
        self.fc1 = nn.Linear(784, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 10)

    def forward(self, x):
        x = self.flatten(x) # (64, 28, 28) -> (64, 784)
        f1 = self.fc1(x) # (64, 784) -> (64, 8)
        h1 = torch.relu(f1) #(64, 8)
        f2 = self.fc2(h1) # (64, 8) -> (64, 8)
        h2 = torch.relu(f2) #(64, 8)
        f3 = self.fc3(h2) # (64, 8) -> (64, 8)
        h3 = torch.relu(f3) #(64, 8)
        logits = self.fc4(h3) #(64, 10)
        return logits #(64, 10)