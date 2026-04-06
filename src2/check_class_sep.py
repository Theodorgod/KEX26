import numpy as np
import torch
from pathlib import Path
import config
from loader import EventDataset
from preprocessing import preprocess_events

print("=" * 70)
print("CHECKING IF EVENTS DIFFER BY CLASS")
print("=" * 70)

ds = EventDataset(split="train", max_files=300)

# Group samples by class
samples_by_class = {0: [], 1: [], 2: []}
for i in range(len(ds)):
    voxel, label = ds[i]
    label_int = label.item()
    samples_by_class[label_int].append(voxel)

# Compute statistics per class
class_names = {0: "paper", 1: "rock", 2: "scissor"}
for class_id in [0, 1, 2]:
    voxels = torch.stack(samples_by_class[class_id])
    
    print(f"\n{class_names[class_id].upper()} (class {class_id}):")
    print(f"  Samples: {len(voxels)}")
    print(f"  Shape: {voxels.shape}")
    print(f"  Mean activity: {(voxels != 0).float().mean():.4f}")
    print(f"  Mean non-zero value: {voxels[voxels != 0].mean():.4f}")
    print(f"  Std non-zero value: {voxels[voxels != 0].std():.4f}")
    
    # Per-channel stats
    for t in range(voxels.shape[1]):
        ch_data = voxels[:, t]  # (N, H, W)
        ch_activity = (ch_data != 0).float().mean().item()
        if ch_activity > 0:
            ch_mean = ch_data[ch_data != 0].mean().item()
        else:
            ch_mean = 0
        print(f"    Time bin {t}: activity={ch_activity:.2%}, mean={ch_mean:.4f}")

# Check class separability
print("\n" + "=" * 70)
print("CLASS SEPARABILITY CHECK")
print("=" * 70)

means = {}
for class_id in [0, 1, 2]:
    voxels = torch.stack(samples_by_class[class_id])
    means[class_id] = voxels.mean(dim=0)
    print(f"\n{class_names[class_id]}: mean shape {means[class_id].shape}")

# Compute distances between class means
print("\nDistance between class mean vectors:")
for i in [0, 1, 2]:
    for j in [i+1, i+2]:
        if j <= 2:
            dist = (means[i] - means[j]).norm().item()
            print(f"  {class_names[i]} <-> {class_names[j]}: {dist:.4f}")

# Check if any one random sample can be distinguished
print("\n" + "=" * 70)
print("SAMPLE DISTINCTIVENESS")
print("=" * 70)

for class_id in [0, 1]:
    sample = samples_by_class[class_id][0]
    distances_to_means = {}
    for other_class in [0, 1, 2]:
        dist = (sample - means[other_class]).norm().item()
        distances_to_means[other_class] = dist
    
    closest_class = min(distances_to_means, key=distances_to_means.get)
    is_correct = closest_class == class_id
    print(f"{class_names[class_id]} sample: closest to {class_names[closest_class]} {'✓' if is_correct else '✗'}")
    for c in [0, 1, 2]:
        print(f"    dist to {class_names[c]}: {distances_to_means[c]:.4f}")
