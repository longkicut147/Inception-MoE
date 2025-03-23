import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------Inception Model----------------------------------------
class InceptionModule(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionModule, self).__init__()
        
        # Path 1: 1x1 Conv
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        
        # Path 2: 1x1 Conv -> 3x3 Conv
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        
        # Path 3: 1x1 Conv -> 5x5 Conv
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        
        # Path 4: 3x3 MaxPool -> 1x1 Conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        p1 = self.relu(self.p1_1(x))
        p2 = self.relu(self.p2_2(self.relu(self.p2_1(x))))
        p3 = self.relu(self.p3_2(self.relu(self.p3_1(x))))
        p4 = self.relu(self.p4_2(self.p4_1(x)))
        
        return torch.cat((p1, p2, p3, p4), dim=1)


class CNN_Inception(nn.Module):
    def __init__(self, in_channels=3, dropout=0.5):
        super(CNN_Inception, self).__init__()

        # 7x7 Conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), # 32,32,32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                # 32,16,16
        )

        # 1x1 Conv -> 3x3 Conv
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),                               # 32,16,16
            nn.ReLU(),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),                    # 96,16,16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                # 96,8,8
        )

        # Inception Module x2
        self.b3 = nn.Sequential(
            InceptionModule(96, 32, (48, 64), (8, 16), 16),                 # 128,8,8
            InceptionModule(128, 64, (64, 96), (16, 48), 48),               # 256,8,8
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                # 256,4,4
        )

        # Inception Module x3
        self.b4 = nn.Sequential(
            InceptionModule(256, 128, (64, 256), (32, 64), 64),             # 512,4,4
            InceptionModule(512, 128, (128, 256), (32, 64), 64),            # 512,4,4
            InceptionModule(512, 256, (160, 320), (32, 128), 128),          # 832,4,4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                # 832,2,2
        )
        
        # Inception Module x2
        self.b5 = nn.Sequential(
            InceptionModule(832, 256, (160, 320), (32, 128), 128),          # 832,2,2
            InceptionModule(832, 384, (192, 384), (48, 128), 128),          # 1024,2,2
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)                # 1024,1,1
        )

        self.fc = nn.Linear(1024, 10)  # 10 classes for output

    def forward(self, x, extract_features=False):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = torch.flatten(x, start_dim=1)
        
        if extract_features:
            return x
        
        return self.fc(x)
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