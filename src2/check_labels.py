import json
from pathlib import Path
import config

data_dir = Path(config.TRAIN_DIR)
labels = json.load(open(config.LABEL_MAP_PATH))
class_to_idx = {name: int(idx) for idx, name in labels.items()}

print("Sample file <-> label mapping:")
print("=" * 70)

all_files = list(data_dir.glob("*_td.dat"))
import random
seed = hash(str("train")) % (2**32)
random.Random(seed).shuffle(all_files)

for dat_path in all_files[:20]:
    class_name_from_file = dat_path.stem.split("_")[0]
    label_from_file = class_to_idx.get(class_name_from_file, "UNKNOWN")
    print(f"{dat_path.name:50} -> class: {class_name_from_file:10} (label={label_from_file})")

print("\n" + "=" * 70)
print("Config values:")
print(f"  DELTA_T: {config.DELTA_T} microseconds ({config.DELTA_T/1e6} seconds)")
print(f"  TIME_BINS: {config.TIME_BINS}")
print(f"  WIDTH: {config.WIDTH}, HEIGHT: {config.HEIGHT}")
print(f"  PREPROCESSING_METHOD: {config.PREPROCESSING_METHOD}")
