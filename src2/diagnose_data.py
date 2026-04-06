import numpy as np
import json
from pathlib import Path
import config
from loader import EventDataset

# Load data
print("=" * 60)
print("DATA DIAGNOSTICS")
print("=" * 60)

train_ds = EventDataset(split="train", max_files=100)
print(f"\nDataset loaded: {len(train_ds)} samples")
print(f"Classes: {train_ds.class_to_index}")

# Check label distribution
labels = []
for i in range(len(train_ds)):
    _, label = train_ds[i]
    labels.append(label.item())

labels = np.array(labels)
unique, counts = np.unique(labels, return_counts=True)
print(f"\nLabel distribution in first 100 samples:")
for u, c in zip(unique, counts):
    class_name = [name for name, idx in train_ds.class_to_index.items() if idx == u][0]
    print(f"  Class {u} ({class_name}): {c} samples ({100*c/len(labels):.1f}%)")

# Check sample quality
print(f"\n" + "="*60)
print("SAMPLE QUALITY CHECK")
print("="*60)

for idx in [0, 1, 2]:
    voxel, label = train_ds[idx]
    class_name = [name for name, idx_lookup in train_ds.class_to_index.items() if idx_lookup == label.item()][0]
    
    print(f"\nSample {idx}:")
    print(f"  Label: {label.item()} ({class_name})")
    print(f"  Shape: {voxel.shape}")
    print(f"  Min: {voxel.min():.4f}, Max: {voxel.max():.4f}, Mean: {voxel.mean():.4f}")
    print(f"  Num zeros: {(voxel == 0).sum().item()} / {voxel.numel()}")
    print(f"  Num non-zeros: {(voxel != 0).sum().item()} / {voxel.numel()}")
    
    if voxel.numel() > 0 and (voxel != 0).sum() > 0:
        non_zero = voxel[voxel != 0]
        print(f"  Non-zero values - Min: {non_zero.min():.4f}, Max: {non_zero.max():.4f}, Mean: {non_zero.mean():.4f}")

print("\n" + "="*60)
print("VALIDATION DATA")
print("="*60)

val_ds = EventDataset(split="val", max_files=160)
print(f"Val dataset loaded: {len(val_ds)} samples")

val_labels = []
for i in range(len(val_ds)):
    _, label = val_ds[i]
    val_labels.append(label.item())

val_labels = np.array(val_labels)
unique, counts = np.unique(val_labels, return_counts=True)
print(f"Val label distribution:")
for u, c in zip(unique, counts):
    class_name = [name for name, idx in val_ds.class_to_index.items() if idx == u][0]
    print(f"  Class {u} ({class_name}): {c} samples ({100*c/len(val_labels):.1f}%)")
