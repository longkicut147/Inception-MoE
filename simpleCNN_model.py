import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)

        self.fc1 = nn.Linear(64*64, 512)
        self.fc2 = nn.Linear(512, 10)
        

    def forward(self, x):               # (batch_size, 3, 32, 32)
        x = self.conv1(x)               # (batch_size, 32, 32, 32)
        x = F.relu(self.batch1(x))      
        x = self.maxpool(x)             # (batch_size, 32, 16, 16)

        x = self.conv2(x)               # (batch_size, 64, 16, 16)
        x = F.relu(self.batch2(x))
        x = self.maxpool(x)             # (batch_size, 64, 8, 8)

        x = x.view(x.size(0), -1)       # (batch_size, 64*8*8)
        x = F.relu(self.fc1(x))         # (batch_size, 512)
        x = self.fc2(x)                 # (batch_size, 10)

        return x