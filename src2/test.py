import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

data_path = Path(__file__).resolve().parent / "test/moorea_2019-01-30_000_td_671500000_731500000_td.dat"

dt = 500000     # microseconds per frame
chunk_size = 1_000_000
max_render_pixels = 4_000_000
num_frames = 60
ncols = 4
view_mode = "animation"  # "grid" or "animation"
animation_interval_ms = 100

frame_accumulators = [dict() for _ in range(num_frames)]


def load_dat_tools_module(openeb_root):
    dat_tools_path = openeb_root / "sdk/modules/core/python/pypkg/metavision_core/event_io/dat_tools.py"
    if not dat_tools_path.exists():
        raise RuntimeError(f"Could not find dat_tools.py at {dat_tools_path}")

    spec = importlib.util.spec_from_file_location("openeb_dat_tools", dat_tools_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_events_vectorized(acc, xs, ys, ps):
    if xs.size == 0:
        return

    keys = (ys.astype(np.uint32) << 16) | xs.astype(np.uint32)
    polarities = np.where(ps > 0, 1.0, -1.0)

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=polarities)

    for key, value in zip(unique_keys, sums):
        acc[int(key)] = acc.get(int(key), 0.0) + float(value)


openeb_root = Path(__file__).resolve().parents[2]
dat_tools = load_dat_tools_module(openeb_root)

with open(data_path, "rb") as f:
    _, ev_type, _, _ = dat_tools.parse_header(f)
    if ev_type not in dat_tools.EV_TYPES:
        raise RuntimeError(f"Unsupported DAT event type: {ev_type}")

    raw_dtype = dat_tools.EV_TYPES[ev_type]

    t_start = None
    frame_end = None

    while True:
        chunk = np.fromfile(f, dtype=raw_dtype, count=chunk_size)
        if chunk.size == 0:
            break

        if "_" in chunk.dtype.names:
            xs = np.bitwise_and(chunk["_"], dat_tools.X_MASK).astype(np.uint16)
            ys = np.right_shift(np.bitwise_and(chunk["_"], dat_tools.Y_MASK), 14).astype(np.uint16)
            ps = np.right_shift(np.bitwise_and(chunk["_"], dat_tools.P_MASK), 28).astype(np.int8)
            ts = chunk["t"].astype(np.int64)
        else:
            xs = chunk["x"].astype(np.uint16)
            ys = chunk["y"].astype(np.uint16)
            ps = chunk["p"].astype(np.int8)
            ts = chunk["t"].astype(np.int64)

        if ts.size == 0:
            continue

        if t_start is None:
            t_start = int(ts[0])
        frame_indices = ((ts - t_start) // dt).astype(np.int64)

        in_range = (frame_indices >= 0) & (frame_indices < num_frames)
        if np.any(in_range):
            used_frame_indices = frame_indices[in_range]
            used_xs = xs[in_range]
            used_ys = ys[in_range]
            used_ps = ps[in_range]

            unique_frames = np.unique(used_frame_indices)
            for fi in unique_frames:
                mask = used_frame_indices == fi
                add_events_vectorized(
                    frame_accumulators[int(fi)],
                    used_xs[mask],
                    used_ys[mask],
                    used_ps[mask],
                )

        if np.any(frame_indices >= num_frames):
            break
    
print("Plotting")
non_empty_frames = [i for i, acc in enumerate(frame_accumulators) if len(acc) > 0]
if not non_empty_frames:
    raise RuntimeError("No events found in the requested frame range")

frame_payloads = []
all_values_for_scale = []
all_xmins, all_xmaxs, all_ymins, all_ymaxs = [], [], [], []

for i in range(num_frames):
    accumulator = frame_accumulators[i]
    if len(accumulator) == 0:
        frame_payloads.append(None)
        continue

    keys = np.fromiter(accumulator.keys(), dtype=np.uint32, count=len(accumulator))
    values = np.fromiter(accumulator.values(), dtype=np.float32, count=len(accumulator))
    ys = (keys >> 16).astype(np.int32)
    xs = (keys & 0xFFFF).astype(np.int32)

    pos_pixels = int(np.sum(values > 0))
    neg_pixels = int(np.sum(values < 0))
    print(f"Frame {i}: active={len(values)}, positive={pos_pixels}, negative={neg_pixels}")

    frame_payloads.append((xs, ys, values))
    all_values_for_scale.append(values)
    all_xmins.append(int(xs.min()))
    all_xmaxs.append(int(xs.max()))
    all_ymins.append(int(ys.min()))
    all_ymaxs.append(int(ys.max()))

all_abs_values = np.abs(np.concatenate(all_values_for_scale))
global_robust_max_abs = max(float(np.percentile(all_abs_values, 99.5)), 1e-6)

if view_mode == "grid":
    nrows = int(np.ceil(num_frames / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i in range(num_frames):
        ax = axes_flat[i]
        payload = frame_payloads[i]
        if payload is None:
            ax.set_title(f"Frame {i}: empty")
            ax.axis("off")
            continue

        xs, ys, values = payload
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        render_h = y_max - y_min + 1
        render_w = x_max - x_min + 1

        if render_h * render_w <= max_render_pixels:
            frame = np.zeros((render_h, render_w), dtype=np.float32)
            frame[ys - y_min, xs - x_min] = values
            frame = np.clip(frame, -global_robust_max_abs, global_robust_max_abs)
            im = ax.imshow(
                frame,
                cmap='bwr',
                vmin=-global_robust_max_abs,
                vmax=global_robust_max_abs,
                interpolation='nearest'
            )
            ax.set_title(f"Frame {i} (dense {render_w}x{render_h})")
        else:
            clipped_values = np.clip(values, -global_robust_max_abs, global_robust_max_abs)
            im = ax.scatter(
                xs,
                ys,
                c=clipped_values,
                cmap='bwr',
                s=2,
                marker='s',
                linewidths=0,
                vmin=-global_robust_max_abs,
                vmax=global_robust_max_abs
            )
            ax.invert_yaxis()
            ax.set_title(f"Frame {i} (sparse {len(values)} px)")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(num_frames, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(min(all_xmins), max(all_xmaxs))
    ax.set_ylim(max(all_ymaxs), min(all_ymins))
    ax.set_title("Frame 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    norm = Normalize(vmin=-global_robust_max_abs, vmax=global_robust_max_abs)
    scatter = ax.scatter([], [], c=[], cmap='bwr', norm=norm, s=2, marker='s', linewidths=0)
    fig.colorbar(scatter, ax=ax, label="Signed event sum")

    def update(frame_idx):
        payload = frame_payloads[frame_idx]
        if payload is None:
            scatter.set_offsets(np.empty((0, 2), dtype=np.float32))
            scatter.set_array(np.array([], dtype=np.float32))
            ax.set_title(f"Frame {frame_idx}: empty")
            return (scatter,)

        xs, ys, values = payload
        clipped_values = np.clip(values, -global_robust_max_abs, global_robust_max_abs)
        offsets = np.column_stack((xs, ys))
        scatter.set_offsets(offsets)
        scatter.set_array(clipped_values)
        ax.set_title(f"Frame {frame_idx}")
        return (scatter,)

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=animation_interval_ms,
        blit=False,
        repeat=True,
    )

plt.show()