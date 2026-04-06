import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual
        return F.relu(x, inplace=True)


class EventCNN(nn.Module):

    def __init__(self, bins, num_classes=10, dropout=0.0, feature_dims=(64, 128, 256), blocks_per_stage=2):

        super().__init__()

        if len(feature_dims) != 3:
            raise ValueError("feature_dims must contain exactly 3 integers, e.g. (64, 128, 256)")

        c1, c2, c3 = [int(v) for v in feature_dims]
        if min(c1, c2, c3) <= 0:
            raise ValueError("feature_dims values must be positive")
        blocks_per_stage = max(1, int(blocks_per_stage))

        self.stem = nn.Sequential(
            nn.Conv2d(bins, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c1, c1, stride=1, blocks=blocks_per_stage)
        self.down1 = nn.MaxPool2d(2)
        self.stage2 = self._make_stage(c1, c2, stride=2, blocks=blocks_per_stage)
        self.stage3 = self._make_stage(c2, c3, stride=2, blocks=blocks_per_stage)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(c3, num_classes),
        )

    def _make_stage(self, in_channels, out_channels, stride, blocks):

        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.dropout(x)

        return self.head(x)