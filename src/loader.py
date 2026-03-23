import torch
from torch.utils.data import Dataset
from metavision_core.event_io import EventsIterator
from preprocessing import events_to_voxel
import config
import json
from pathlib import Path


class EventDataset(Dataset):

    def __init__(self, split="train", max_files=None):

        split_dirs = {
            "train": config.TRAIN_DIR,
            "val": config.VAL_DIR,
        }
        if split not in split_dirs:
            raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'val'.")

        data_dir = Path(split_dirs[split])
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

        with open(config.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        # Convert {"0": "paper"} into {"paper": 0} for fast lookup.
        self.class_to_index = {name: int(idx) for idx, name in label_map.items()}

        self.samples = []
        for dat_path in sorted(data_dir.glob("*_td.dat")):
            bbox_path = dat_path.with_name(dat_path.stem.replace("_td", "_bbox") + ".npy")
            if not bbox_path.exists():
                continue

            class_name = dat_path.stem.split("_")[0]
            if class_name not in self.class_to_index:
                continue

            self.samples.append(
                {
                    "dat_path": dat_path,
                    "bbox_path": bbox_path,
                    "label": self.class_to_index[class_name],
                }
            )

        if max_files is None:
            max_files = config.MAX_TRAIN_FILES
        if max_files is not None:
            self.samples = self.samples[: max(0, int(max_files))]

        if not self.samples:
            raise RuntimeError(
                f"No usable samples found in {data_dir}. "
                "Expected matching '*_td.dat' and '*_bbox.npy' files with labels from label_map_dictionary.json."
            )

    def _first_event_slice(self, dat_path):
        iterator = EventsIterator(
            input_path=str(dat_path),
            delta_t=config.DELTA_T,
        )
        for evs in iterator:
            return evs
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        evs = self._first_event_slice(sample["dat_path"])

        voxel = events_to_voxel(
            evs,
            config.HEIGHT,
            config.WIDTH,
            config.TIME_BINS
        )

        voxel = torch.tensor(voxel, dtype=torch.float32)

        label = torch.tensor(sample["label"], dtype=torch.long)

        return voxel, label