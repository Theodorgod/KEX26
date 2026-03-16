import numpy as np

def events_to_voxel(events, H, W, T):

    voxel = np.zeros((T, H, W), dtype=np.float32)

    if len(events) == 0:
        return voxel

    t = events["t"]
    x = events["x"] // 4
    y = events["y"] // 4
    p = events["p"]

    t0 = t[0]
    t1 = t[-1]

    dt = t1 - t0 + 1

    bins = ((t - t0) * (T - 1) / dt).astype(np.int32)

    polarity = np.where(p > 0, 1, -1)

    valid = (x < W) & (y < H)

    np.add.at(voxel, (bins[valid], y[valid], x[valid]), polarity[valid])

    return voxel