import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from loader import EventDataset
from model import EventCNN
import config

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = EventDataset()

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

model = EventCNN(config.TIME_BINS).to(device)

optimizer = optim.Adam(model.parameters(), lr=config.LR)

criterion = nn.CrossEntropyLoss()

for epoch in range(config.EPOCHS):

    for voxels, labels in loader:
        print("Starting training step")
        voxels = voxels.to(device)
        labels = labels.to(device)

        pred = model(voxels)

        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch", epoch, "loss", loss.item())

torch.save(model.state_dict(), "model.pth")