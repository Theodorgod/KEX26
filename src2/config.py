import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("OPENEB_EVENT_DATA_DIR", str(ROOT_DIR / "data")))

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
LABEL_MAP_PATH = DATA_DIR / "label_map_dictionary.json"

WIDTH = 320
HEIGHT = 180

TIME_BINS = 12

# Preprocessing options: "event_frame", "polarity_frame", "time_bins", "tbr", "time_surface"
PREPROCESSING_METHOD = "time_bins"
DOWNSAMPLE_FACTOR = 4
INPUT_NORMALIZATION = "none"
TIME_SURFACE_DECAY = 3.0  # exponential decay lambda for time_surface method

DELTA_T = 500000

# How to select the time window from each recording:
#   "full"        – use every event in the file (guaranteed gesture, adapts to length)
#   "first_slice" – first DELTA_T microseconds (original, fast)
#   "dense"       – DELTA_T window with the most events (focuses on peak activity)
#   "bbox"        – use only the bbox-annotated timestamp interval
#   "active_slice"– scan fixed windows and pick the most active one
#   "random"      – random DELTA_T window each call (data-augmentation, varies per epoch)
WINDOW_MODE = "full"
BBOX_JITTER_US = 0
ACTIVE_SLICE_STRIDE_US = 100000
ACTIVE_SLICE_TOP_K = 1
CACHE_PREPROCESSED = False
PREPROCESSED_CACHE_DIRNAME = ".core_ml_cache"

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3

CPU_COUNT = os.cpu_count() or 8
NUM_WORKERS = min(12, max(4, CPU_COUNT // 2))
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
DATASET_SEED = 1337

# Optional limits useful for smoke tests and debugging.
MAX_TRAIN_FILES = None

DEVICE = "cuda"