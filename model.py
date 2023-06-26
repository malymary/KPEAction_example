import torch.nn as nn


class EnDeCoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 56)

        self.l5 = nn.Linear(56, 128)
        self.l6 = nn.Linear(128, 256)
        self.l7 = nn.Linear(256, 512)
        self.l8 = nn.Linear(512, 784)

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.l5(x)
        x = self.relu(x)
        x = self.l6(x)
        x = self.relu(x)
        x = self.l7(x)
        x = self.relu(x)
        x = self.l8(x)
        return x

