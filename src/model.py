import torch.nn as nn
import torch.nn.functional as F


class EventCNN(nn.Module):

    def __init__(self, bins):

        super().__init__()

        self.conv1 = nn.Conv2d(bins, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc = nn.Linear(128 * 22 * 40, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.flatten(1)

        return self.fc(x)