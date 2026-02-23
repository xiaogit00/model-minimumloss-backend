class CNNNet(nn.Module):
    """
    A simple CNN with 1 convolutional layer of kernel 8, stride 2, and no maxpooling.

    Dimension sizes:
    D_i: (64, 1, 28, 28)
    (W-F+2P)/S + 1 = (28-4+0)/2+1 = 13
    D_1: (64, 8, 13, 13)
    D_O: (8*13*13, 10)

    """
    def __init__(self, num_classes=2):
        super(CNNNet, self).__init__()
        self.cnn1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 13 * 13, 10) 


    def forward(self, x):
        f1 = self.cnn1(x)
        h1 = self.relu(f1)
        h1_flattened = torch.flatten(h1, 1)
        logits = self.fc1(h1_flattened)

        return logits