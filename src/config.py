from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
LABEL_MAP_PATH = DATA_DIR / "label_map_dictionary.json"

WIDTH = 320
HEIGHT = 180

TIME_BINS = 5

DELTA_T = 200000

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3

NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Optional limits useful for smoke tests and debugging.
MAX_TRAIN_FILES = None

DEVICE = "cuda"