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