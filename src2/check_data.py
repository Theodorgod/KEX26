import config, json
from pathlib import Path
import os

vd = Path(config.VAL_DIR)
print(f"VAL_DIR: {vd}")
print(f"Exists: {vd.exists()}")

if vd.exists():
    samples = [p for p in vd.glob("*_td.dat") if p.with_name(p.stem.replace("_td", "_bbox") + ".npy").exists()]
    print(f"Val samples available: {len(samples)}")
else:
    print("Val directory does not exist - using train/val split from single directory")

td = Path(config.TRAIN_DIR) 
if td.exists():
    samples = [p for p in td.glob("*_td.dat") if p.with_name(p.stem.replace("_td", "_bbox") + ".npy").exists()]
    print(f"Train samples found: {len(samples)}")
    
    if os.path.exists(config.LABEL_MAP_PATH):
        labels = json.load(open(config.LABEL_MAP_PATH))
        print(f"Classes: {labels}")
        print(f"Num classes: {len(labels)}")
    
    # Check class distribution
    class_to_idx = {name: int(idx) for idx, name in labels.items()}
    class_counts = {name: 0 for name in labels.values()}
    for p in samples:
        class_name = p.stem.split("_")[0]
        if class_name in class_to_idx.keys():
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nClass distribution:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")
