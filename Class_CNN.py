# Class_CNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.proj  = nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False) if in_ch != out_ch or stride != 1 else None

    def forward(self, x):
        idn = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        if self.proj is not None:
            idn = self.proj(idn)
        return F.relu(x + idn, inplace=True)

class SmallResNet3D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 3):
        super().__init__()
        ch = [32, 64, 128, 256]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(ch[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = BasicBlock3D(ch[0], ch[0])
        self.stage2 = BasicBlock3D(ch[0], ch[1], stride=2)  # downsample
        self.stage3 = BasicBlock3D(ch[1], ch[2], stride=2)
        self.stage4 = BasicBlock3D(ch[2], ch[3], stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch[3], num_classes)
        )

    def forward(self, x):  # x: [B, C, D, H, W]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)
