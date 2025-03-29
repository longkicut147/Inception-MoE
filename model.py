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

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # 5x5 Conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2), # 32,32,32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 1x1 Conv -> 3x3 Conv
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),                             # 32,16,16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                    # 64,16,16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Inception Module x2
        self.b3 = nn.Sequential(
            InceptionModule(64, 32, (48, 64), (16, 32), 16),                # 144,8,8
        )

        self.b4 = nn.Sequential(
            InceptionModule(144, 48, (64, 96), (16, 32), 16),               # 192,4,4
            InceptionModule(192, 64, (80, 128), (24, 48), 24),              # 264,4,4
        )

        self.b5 = nn.Sequential(
            InceptionModule(264, 64, (80, 112), (32, 40), 40),              # 256,2,2
        )

        self.fc = nn.Linear(256, 10)  # 10 classes for output

    def forward(self, x, extract_features=False):
        x = self.b1(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.b2(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.b3(x)
        x = self.maxpool(x)
        x = self.b4(x)
        x = self.maxpool(x)

        x = self.b5(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        
        if extract_features:
            return x
        
        return self.fc(x)
#----------------------------------------Inception Model----------------------------------------


#----------------------------------------Simple CNN Model---------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn1d = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x, extract_features=False):
        x = F.relu(self.bn1(self.conv1(x))) # 32x32x32
        x = self.maxpool(x)                 # 32x16x16
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x))) # 64x16x16
        x = self.maxpool(x)                 # 64x8x8
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x))) # 128x8x8
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
    def __init__(self, num_classes=10, dropout=0.5):
        super(CNN_Resnet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(dropout)

        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, downsample=True)
        self.layer3 = ResidualBlock(64, 128, downsample=True)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, extract_features=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

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


#----------------------------------------Inception Gating---------------------------------------
class Inception_Gating_Module(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, top_k=2):
        super(Inception_Gating_Module, self).__init__()
        
        self.top_k = top_k  # Số path quan trọng nhất cần giữ lại
        
        # Path 1: 1x1 Conv
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p1_bn = nn.BatchNorm2d(c1)
        self.p1_fc = nn.Conv2d(c1, 128, kernel_size=1)

        # Path 2: 1x1 Conv -> 3x3 Conv
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p2_bn = nn.BatchNorm2d(c2[1])
        self.p2_fc = nn.Conv2d(c2[1], 128, kernel_size=1)

        # Path 3: 1x1 Conv -> 5x5 Conv
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p3_bn = nn.BatchNorm2d(c3[1])
        self.p3_fc = nn.Conv2d(c3[1], 128, kernel_size=1)

        # Path 4: 3x3 MaxPool -> 1x1 Conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        self.p4_bn = nn.BatchNorm2d(c4)
        self.p4_fc = nn.Conv2d(c4, 128, kernel_size=1)
        
        self.relu = nn.ReLU()

        # Gating Network 
        self.gating = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  
            nn.Linear(32, 4),  
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        # Get gating output
        gate_output = self.gating(x)  # (batch_size, 4)

        # Sparse activation (select top-k experts)
        topk_values, topk_indices = torch.topk(gate_output, self.top_k, dim=1)

        # Let top-k path = 1, else (not use) = 0
        mask = torch.zeros_like(gate_output)
        mask.scatter_(1, topk_indices, 1)

        # So that weights of top-k path is 1*weight = weight, else 0*weight = 0 (not use)
        gate_weights = gate_output * mask

        # Normalize the weights so that the sum = 1
        gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True)

        # Paths' output * gate weights
        p1 = self.relu(self.p1_bn(self.p1_1(x))) 
        p1 = self.p1_fc(p1)
        p1 = p1 * gate_weights[:, 0].view(-1, 1, 1, 1)

        p2 = self.relu(self.p2_bn(self.p2_2(self.relu(self.p2_1(x))))) 
        p2 = self.p2_fc(p2)
        p2 = p2 * gate_weights[:, 1].view(-1, 1, 1, 1)

        p3 = self.relu(self.p3_bn(self.p3_2(self.relu(self.p3_1(x)))))
        p3 = self.p3_fc(p3)
        p3 = p3 * gate_weights[:, 2].view(-1, 1, 1, 1)

        p4 = self.relu(self.p4_bn(self.p4_2(self.p4_1(x))))
        p4 = self.p4_fc(p4)
        p4 = p4 * gate_weights[:, 3].view(-1, 1, 1, 1)

        return p1+p2+p3+p4

class CNN_Inception_Gating(nn.Module):
    def __init__(self, in_channels=3, dropout=0.5):
        super(CNN_Inception_Gating, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # 5x5 Conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2), # 32,32,32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 1x1 Conv -> 3x3 Conv
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),                             # 32,16,16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                    # 64,16,16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Inception Module x2
        self.b3 = nn.Sequential(
            Inception_Gating_Module(64, 32, (48, 64), (16, 32), 16),                # 144,8,8
        )

        self.b4 = nn.Sequential(
            Inception_Gating_Module(128, 48, (64, 96), (16, 32), 16),               # 192,4,4
            Inception_Gating_Module(128, 64, (80, 128), (24, 48), 24),              # 264,4,4
        )

        self.b5 = nn.Sequential(
            Inception_Gating_Module(128, 64, (80, 112), (32, 40), 40),              # 256,2,2
        )

        self.fc = nn.Linear(128, 10)  # 10 classes for output

    def forward(self, x, extract_features=False):
        x = self.b1(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.b2(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.b3(x)
        x = self.maxpool(x)
        x = self.b4(x)
        x = self.maxpool(x)

        x = self.b5(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        
        if extract_features:
            return x
        
        return self.fc(x)