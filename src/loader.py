import torch
from torch.utils.data import Dataset
from metavision_core.event_io import EventsIterator
from preprocessing import events_to_voxel
import config


class EventDataset(Dataset):

    def __init__(self):

        self.slices = []

        iterator = EventsIterator(
            input_path=config.DATA_PATH,
            delta_t=config.DELTA_T
        )

        for evs in iterator:
            self.slices.append(evs)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        evs = self.slices[idx]

        voxel = events_to_voxel(
            evs,
            config.HEIGHT,
            config.WIDTH,
            config.TIME_BINS
        )

        voxel = torch.tensor(voxel, dtype=torch.float32)

        label = torch.randint(0, 10, (1,)).item()

        return voxel, label