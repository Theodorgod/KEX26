import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

OPENEB_CORE_PYPKG = Path(__file__).resolve().parents[3] / "core/python/pypkg"
if str(OPENEB_CORE_PYPKG) not in sys.path:
    sys.path.insert(0, str(OPENEB_CORE_PYPKG))

from metavision_core.event_io.py_reader import EventDatReader

import config
from preprocessing import preprocess_events, get_num_channels


class EventDataset(Dataset):

    def __init__(self, split="train", max_files=None):
        self.split = split
        self.is_train = split == "train"
        self.preprocessing_method = config.PREPROCESSING_METHOD
        self.time_bins = int(config.TIME_BINS)
        self.height = int(config.HEIGHT)
        self.width = int(config.WIDTH)
        self.downsample_factor = int(config.DOWNSAMPLE_FACTOR)
        self.delta_t = int(config.DELTA_T)
        self.input_normalization = str(config.INPUT_NORMALIZATION)
        self.time_surface_decay = float(config.TIME_SURFACE_DECAY)
        self.window_mode = config.WINDOW_MODE
        self.bbox_jitter_us = int(config.BBOX_JITTER_US) if self.is_train else 0
        self.active_slice_stride_us = max(1, int(config.ACTIVE_SLICE_STRIDE_US))
        self.active_slice_top_k = max(1, int(config.ACTIVE_SLICE_TOP_K))
        self.num_channels = get_num_channels(self.preprocessing_method, self.time_bins)
        self.cache_preprocessed = bool(
            config.CACHE_PREPROCESSED
            and self.window_mode != "random"
            and not (self.window_mode == "bbox" and self.bbox_jitter_us > 0)
            and not (self.window_mode == "active_slice" and self.is_train and self.active_slice_top_k > 1)
        )

        split_dirs = {
            "train": config.TRAIN_DIR,
            "val": config.VAL_DIR,
        }
        if split not in split_dirs:
            raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'val'.")

        data_dir = Path(split_dirs[split])
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
        self.cache_dir = data_dir / config.PREPROCESSED_CACHE_DIRNAME

        with open(config.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        # Convert {"0": "paper"} into {"paper": 0} for fast lookup.
        self.class_to_index = {name: int(idx) for idx, name in label_map.items()}

        self.samples = []
        all_files = list(data_dir.glob("*_td.dat"))
        
        # Shuffle with stable explicit seed so method comparisons are reproducible.
        import random
        seed_base = int(config.DATASET_SEED)
        split_offset = 0 if split == "train" else 1
        random.Random(seed_base + split_offset).shuffle(all_files)
        
        for dat_path in all_files:
            bbox_path = dat_path.with_name(dat_path.stem.replace("_td", "_bbox") + ".npy")
            if not bbox_path.exists():
                continue

            bbox_start_ts = None
            bbox_end_ts = None
            try:
                bbox_data = np.load(bbox_path, mmap_mode="r")
                if len(bbox_data):
                    bbox_start_ts = int(bbox_data["ts"][0])
                    bbox_end_ts = int(bbox_data["ts"][-1])
            except Exception:
                bbox_start_ts = None
                bbox_end_ts = None

            class_name = dat_path.stem.split("_")[0]
            if class_name not in self.class_to_index:
                continue

            self.samples.append(
                {
                    "dat_path": dat_path,
                    "bbox_path": bbox_path,
                    "bbox_start_ts": bbox_start_ts,
                    "bbox_end_ts": bbox_end_ts,
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

    def _cache_path(self, dat_path):
        cache_key = (
            f"{self.preprocessing_method}"
            f"_wb{self.window_mode}"
            f"_tb{self.time_bins}"
            f"_ds{self.downsample_factor}"
            f"_dt{self.delta_t}"
            f"_in{self.input_normalization}"
            f"_tsd{self.time_surface_decay}"
            f"_bj{self.bbox_jitter_us}"
            f"_as{self.active_slice_stride_us}"
            f"_atk{self.active_slice_top_k}"
        )
        return self.cache_dir / cache_key / f"{dat_path.stem}.npy"

    def _load_active_slice(self, reader):
        import random as _random

        n = reader.event_count()
        if n == 0:
            return reader.load_delta_t(1)

        all_events = reader.load_n_events(n)
        t = all_events["t"].astype(np.int64)
        if len(t) == 0:
            return all_events

        t0 = int(t[0])
        t1 = int(t[-1])
        if t1 - t0 <= self.delta_t:
            return all_events

        starts = np.arange(t0, t1 - self.delta_t + 1, self.active_slice_stride_us, dtype=np.int64)
        if starts.size == 0:
            starts = np.array([t0], dtype=np.int64)

        ends = starts + self.delta_t
        left = np.searchsorted(t, starts, side="left")
        right = np.searchsorted(t, ends, side="left")
        counts = right - left

        ranked = np.argsort(-counts)
        k = min(self.active_slice_top_k, len(ranked))
        top = ranked[:k]
        if self.is_train and k > 1:
            chosen_idx = int(_random.choice(top.tolist()))
        else:
            chosen_idx = int(top[0])

        start_t = int(starts[chosen_idx])
        reader.seek_time(start_t)
        return reader.load_delta_t(self.delta_t)

    def _preprocess_sample(self, sample):
        evs = self._load_events(sample)
        return preprocess_events(
            evs,
            self.height,
            self.width,
            self.time_bins,
            method=self.preprocessing_method,
            downsample_factor=self.downsample_factor,
            input_normalization=self.input_normalization,
            time_surface_decay=self.time_surface_decay,
        )

    def _load_or_build_voxel(self, sample):
        if not self.cache_preprocessed:
            return self._preprocess_sample(sample)

        cache_path = self._cache_path(sample["dat_path"])
        if cache_path.exists():
            cached = np.load(cache_path)
            expected_shape = (self.num_channels, self.height, self.width)
            if cached.shape == expected_shape:
                return cached
            cache_path.unlink(missing_ok=True)

        voxel = self._preprocess_sample(sample)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp_path, voxel)
        os.replace(tmp_path, cache_path)
        return voxel

    def _load_events(self, sample):
        """Load events from a .dat file according to config.WINDOW_MODE."""
        mode = self.window_mode
        dat_path = sample["dat_path"]
        reader = EventDatReader(str(dat_path))

        if mode == "first_slice":
            return reader.load_delta_t(self.delta_t)

        if mode == "full":
            n = reader.event_count()
            if n == 0:
                return reader.load_delta_t(1)
            return reader.load_n_events(n)

        if mode == "bbox":
            import random as _random
            bbox_start = sample.get("bbox_start_ts")
            bbox_end = sample.get("bbox_end_ts")
            if bbox_start is None or bbox_end is None or bbox_end <= bbox_start:
                n = reader.event_count()
                if n == 0:
                    return reader.load_delta_t(1)
                return reader.load_n_events(n)
            if self.bbox_jitter_us > 0:
                duration = int(bbox_end - bbox_start)
                offset = _random.randint(-self.bbox_jitter_us, self.bbox_jitter_us)
                first_t = int(reader.get_first_ev_timestamp())
                last_t = int(reader.get_last_ev_timestamp())
                bbox_start = max(first_t, min(last_t, int(bbox_start + offset)))
                bbox_end = min(last_t, int(bbox_start + duration))
                if bbox_end <= bbox_start:
                    bbox_end = min(last_t, bbox_start + 1)
            reader.seek_time(bbox_start)
            return reader.load_delta_t(int(bbox_end - bbox_start + 1))

        if mode == "dense":
            import numpy as _np
            n = reader.event_count()
            if n == 0:
                return reader.load_delta_t(1)
            all_events = reader.load_n_events(n)
            t = all_events["t"].astype(_np.int64)
            if len(t) == 0 or int(t[-1]) - int(t[0]) <= self.delta_t:
                return all_events
            # Vectorised sliding-window: for each start event i, find the last
            # event j such that t[j] < t[i] + DELTA_T, then count = j - i.
            end_indices = _np.searchsorted(t, t + self.delta_t, side="left")
            counts = end_indices - _np.arange(len(t))
            best_i = int(_np.argmax(counts))
            t_win = int(t[best_i])
            mask = (t >= t_win) & (t < t_win + self.delta_t)
            return all_events[mask]

        if mode == "active_slice":
            return self._load_active_slice(reader)

        if mode == "random":
            import random as _random
            first_t = reader.get_first_ev_timestamp()
            last_t = reader.get_last_ev_timestamp()
            if last_t - first_t <= self.delta_t:
                reader.reset()
                n = reader.event_count()
                if n == 0:
                    return reader.load_delta_t(1)
                return reader.load_n_events(n)
            max_start = last_t - self.delta_t
            start_t = _random.randint(int(first_t), int(max_start))
            reader.seek_time(start_t)
            return reader.load_delta_t(self.delta_t)

        raise ValueError(
            f"Unknown WINDOW_MODE {mode!r}. "
            "Choose from: 'full', 'first_slice', 'dense', 'bbox', 'active_slice', 'random'."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        voxel = self._load_or_build_voxel(sample)

        voxel = torch.tensor(voxel, dtype=torch.float32)

        label = torch.tensor(sample["label"], dtype=torch.long)

        return voxel, label