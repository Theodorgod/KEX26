import torch
from torch.utils.data import DataLoader
from loader import EventDataset
from model import EventCNN
import config

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = EventDataset()

loader = DataLoader(dataset, batch_size=8)

model = EventCNN(config.TIME_BINS).to(device)
model.load_state_dict(torch.load("model.pth"))

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for voxels, labels in loader:

        voxels = voxels.to(device)
        labels = labels.to(device)

        pred = model(voxels)

        predicted = pred.argmax(1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print("accuracy:", correct / total)