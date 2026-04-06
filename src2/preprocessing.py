import numpy as np


def _normalize_tensor(tensor, mode="none", eps=1e-6):
    mode = (mode or "none").lower()
    if mode == "none":
        return tensor

    if mode == "maxabs":
        scale = float(np.max(np.abs(tensor)))
        if scale < eps:
            return tensor
        return tensor / scale

    if mode == "zscore":
        mean = float(np.mean(tensor))
        std = float(np.std(tensor))
        if std < eps:
            return tensor - mean
        return (tensor - mean) / std

    raise ValueError(f"Unsupported normalization mode: {mode}")

def _downsample_and_mask(events, H, W, downsample_factor):
    x = events["x"] // max(1, int(downsample_factor))
    y = events["y"] // max(1, int(downsample_factor))
    p = events["p"]
    t = events["t"]

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return t[valid], x[valid], y[valid], p[valid]


def events_to_frame(events, H, W, downsample_factor=4):
    frame = np.zeros((1, H, W), dtype=np.float32)

    if len(events) == 0:
        return frame

    _, x, y, p = _downsample_and_mask(events, H, W, downsample_factor)
    if x.size == 0:
        return frame

    polarity = np.where(p > 0, 1.0, -1.0)
    np.add.at(frame[0], (y, x), polarity)
    return frame


def events_to_voxel(events, H, W, T, downsample_factor=4):

    voxel = np.zeros((T, H, W), dtype=np.float32)

    if len(events) == 0:
        return voxel

    t, x, y, p = _downsample_and_mask(events, H, W, downsample_factor)
    if x.size == 0:
        return voxel

    t0 = t[0]
    t1 = t[-1]

    dt = t1 - t0 + 1

    bins = ((t - t0) * (T - 1) / dt).astype(np.int32)

    polarity = np.where(p > 0, 1.0, -1.0)

    np.add.at(voxel, (bins, y, x), polarity)

    return voxel


def events_to_tbr(events, H, W, downsample_factor=4):
    # TBR = signed, timestamp-normalized single-frame encoding.
    tbr = np.zeros((1, H, W), dtype=np.float32)

    if len(events) == 0:
        return tbr

    t, x, y, p = _downsample_and_mask(events, H, W, downsample_factor)
    if x.size == 0:
        return tbr

    t = t.astype(np.float32)
    t0 = t.min()
    t1 = t.max()
    dt = max(1.0, t1 - t0)
    t_norm = (t - t0) / dt

    pos = p > 0
    neg = ~pos

    pos_frame = np.zeros((H, W), dtype=np.float32)
    neg_frame = np.zeros((H, W), dtype=np.float32)

    if np.any(pos):
        np.maximum.at(pos_frame, (y[pos], x[pos]), t_norm[pos])
    if np.any(neg):
        np.maximum.at(neg_frame, (y[neg], x[neg]), t_norm[neg])

    tbr[0] = pos_frame - neg_frame

    return tbr


def events_to_time_surface(events, H, W, downsample_factor=4, decay=3.0):
    # 3-channel representation: [pos_SAE, neg_SAE, log_count_norm]
    #   ch0/1: exponential decay time surface per polarity.
    #     exp(-decay*(t_max - t_last)/dt): recent event → 1.0, no event → ~0.
    #   ch2: log(1 + total_event_count) normalised by log(1 + max_count).
    #     This provides spatial density (WHERE events are) which the SAE
    #     alone discards — key for distinguishing gestures.
    surface = np.zeros((3, H, W), dtype=np.float32)

    if len(events) == 0:
        return surface

    t, x, y, p = _downsample_and_mask(events, H, W, downsample_factor)
    if x.size == 0:
        return surface

    t = t.astype(np.float64)
    t0 = float(t.min())
    t1 = float(t.max())
    dt = max(1.0, t1 - t0)

    pos = p > 0
    neg = ~pos

    # Initialize last-event timestamps to one window before t0 so that empty
    # pixels decay to exp(-2*decay) ≈ 0.0 instead of being truly ambiguous.
    pos_last = np.full((H, W), fill_value=t0 - dt, dtype=np.float64)
    neg_last = np.full((H, W), fill_value=t0 - dt, dtype=np.float64)

    if np.any(pos):
        np.maximum.at(pos_last, (y[pos], x[pos]), t[pos])
    if np.any(neg):
        np.maximum.at(neg_last, (y[neg], x[neg]), t[neg])

    surface[0] = np.exp(-decay * (t1 - pos_last) / dt).astype(np.float32)
    surface[1] = np.exp(-decay * (t1 - neg_last) / dt).astype(np.float32)

    # Count channel: log-normalised total event density per pixel.
    count = np.zeros((H, W), dtype=np.float32)
    np.add.at(count, (y, x), 1.0)
    log_count = np.log1p(count)
    max_log = float(log_count.max())
    if max_log > 0:
        surface[2] = (log_count / max_log).astype(np.float32)

    return surface


def events_to_polarity_frame(events, H, W, downsample_factor=4):
    # 2-channel polarity-split event frame: [pos_count, neg_count].
    # Strictly more informative than the signed 1-channel event_frame because
    # the model can see each polarity's spatial distribution independently.
    frame = np.zeros((2, H, W), dtype=np.float32)

    if len(events) == 0:
        return frame

    _, x, y, p = _downsample_and_mask(events, H, W, downsample_factor)
    if x.size == 0:
        return frame

    pos = p > 0
    neg = ~pos
    if np.any(pos):
        np.add.at(frame[0], (y[pos], x[pos]), 1.0)
    if np.any(neg):
        np.add.at(frame[1], (y[neg], x[neg]), 1.0)

    return frame


def get_num_channels(method, time_bins):
    if method == "time_bins":
        return int(time_bins)
    if method == "time_surface":
        return 3
    if method == "polarity_frame":
        return 2
    return 1


def preprocess_events(events, H, W, time_bins, method="time_bins", downsample_factor=4,
                      input_normalization="none", time_surface_decay=3.0):
    if method == "event_frame":
        out = events_to_frame(events, H, W, downsample_factor=downsample_factor)
        return _normalize_tensor(out, mode=input_normalization)
    if method == "polarity_frame":
        out = events_to_polarity_frame(events, H, W, downsample_factor=downsample_factor)
        return _normalize_tensor(out, mode=input_normalization)
    if method == "time_bins":
        out = events_to_voxel(events, H, W, time_bins, downsample_factor=downsample_factor)
        return _normalize_tensor(out, mode=input_normalization)
    if method == "tbr":
        out = events_to_tbr(events, H, W, downsample_factor=downsample_factor)
        return _normalize_tensor(out, mode=input_normalization)
    if method == "time_surface":
        out = events_to_time_surface(events, H, W, downsample_factor=downsample_factor,
                                     decay=float(time_surface_decay))
        return _normalize_tensor(out, mode=input_normalization)
    raise ValueError(f"Unsupported preprocessing method: {method}")