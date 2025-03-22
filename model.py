import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------Inception Model----------------------------------------
class InceptionModule(nn.Module):
    def __init__(self, in_channels=3):
        super(InceptionModule, self).__init__()

        # Branch 1: 1x1 Convolution
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Branch 2: 1x1 Convolution -> 3x3 Convolution
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Branch 3: 1x1 Convolution -> 5x5 Convolution
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Branch 4: MaxPooling -> 1x1 Convolution
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch3x3(x)
        out3 = self.branch5x5(x)
        out4 = self.branch_pool(x)
        outputs = torch.cat([out1, out2, out3, out4], dim=1)
        return outputs


class StackedInception(nn.Module):
    def __init__(self, input_channels, num_modules=3):
        super(StackedInception, self).__init__()
        ouput_channels = 256
        self.inception_modules = nn.ModuleList([
            InceptionModule(input_channels if i == 0 else ouput_channels) for i in range(num_modules)
        ])

    def forward(self, x):
        for module in self.inception_modules:
            x = module(x)
        return x


class CNN_Inception(nn.Module):
    def __init__(self, in_channels=3):
        super(CNN_Inception, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)

        self.stack1 = StackedInception(192, 2)
        self.stack2 = StackedInception(256, 5)
        self.stack3 = StackedInception(256, 2)

        # Fully Connected to predict output labels
        self.fc = nn.Linear(256, 10)


    def forward(self, x, extract_features=False):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.stack2(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.stack3(x)
        x = self.dropout(x)
        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)
        if extract_features:
            return x
        
        x = self.fc(x)
        return x
#----------------------------------------Inception Model----------------------------------------


#----------------------------------------Simple CNN Model---------------------------------------
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
#----------------------------------------Simple CNN Model---------------------------------------


#----------------------------------------ResNet Model-------------------------------------------