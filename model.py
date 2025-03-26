import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------Inception Model----------------------------------------
class InceptionModule(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionModule, self).__init__()
        
        # Path 1: 1x1 Conv
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p1_bn = nn.BatchNorm2d(c1)

        # Path 2: 1x1 Conv -> 3x3 Conv
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p2_bn = nn.BatchNorm2d(c2[1])

        # Path 3: 1x1 Conv -> 5x5 Conv
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p3_bn = nn.BatchNorm2d(c3[1])

        # Path 4: 3x3 MaxPool -> 1x1 Conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        self.p4_bn = nn.BatchNorm2d(c4)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        p1 = self.relu(self.p1_bn(self.p1_1(x)))
        p2 = self.relu(self.p2_bn(self.p2_2(self.relu(self.p2_1(x)))))
        p3 = self.relu(self.p3_bn(self.p3_2(self.relu(self.p3_1(x)))))
        p4 = self.relu(self.p4_bn(self.p4_2(self.p4_1(x))))
        
        return torch.cat((p1, p2, p3, p4), dim=1)


class CNN_Inception(nn.Module):
    def __init__(self, in_channels=3, dropout=0.5):
        super(CNN_Inception, self).__init__()

        # 7x7 Conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2), # 32,32,32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                # 32,16,16
        )

        # 1x1 Conv -> 3x3 Conv
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),                             # 32,16,16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                    # 64,16,16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                # 64,8,8
        )

        # Inception Module x2
        self.b3 = nn.Sequential(
            InceptionModule(64, 32, (48, 64), (16, 32), 16),                # 144,8,8
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                # 144,4,4
        )

        self.b4 = nn.Sequential(
            InceptionModule(144, 48, (64, 96), (16, 32), 16),               # 192,4,4
            InceptionModule(192, 64, (80, 128), (24, 48), 24),              # 264,4,4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                # 264,2,2
        )

        self.b5 = nn.Sequential(
            InceptionModule(264, 64, (80, 112), (32, 40), 40),              # 256,2,2
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)                # 256,1,1
        )

        self.fc = nn.Linear(256, 10)  # 10 classes for output

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
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn1d = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x, extract_features=False):
        x = F.relu(self.bn1(self.conv1(x))) # 32x32x32
        # x = F.relu(self.bn1(self.conv2(x))) # 32x32x32
        x = self.maxpool(x)                 # 32x16x16
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv3(x))) # 64x16x16
        # x = F.relu(self.bn2(self.conv4(x))) # 64x16x16
        x = self.maxpool(x)                 # 64x8x8
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv5(x))) # 128x8x8
        # x = F.relu(self.bn3(self.conv6(x))) # 128x8x8
        x = self.avgpool(x)                 # 128x4x4
        x = self.dropout(x)
        
        x = torch.flatten(x, 1)             # 128*4*4
        x = F.relu(self.bn1d(self.fc1(x)))
        x = self.dropout(x)

        if extract_features:
            return x
        
        return self.fc2(x)
#----------------------------------------Simple CNN Model---------------------------------------


#----------------------------------------ResNet Model-------------------------------------------
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()

        stride = 2 if downsample else 1  # Nếu downsample, giảm kích thước ảnh
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Tầng shortcut nếu số channels thay đổi
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class CNN_Resnet(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Resnet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)

        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, downsample=True)
        self.layer3 = ResidualBlock(64, 128, downsample=True)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, extract_features=False):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        if extract_features:
            return x

        return self.fc(x)
#----------------------------------------ResNet Model-------------------------------------------
