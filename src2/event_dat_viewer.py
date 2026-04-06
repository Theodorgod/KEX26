import numpy as np
import cv2

FILE = "test/moorea_2019-01-30_000_td_671500000_731500000_td.dat"

WIDTH = 1280
HEIGHT = 720

dtype = np.dtype([
    ("t", np.uint64),
    ("x", np.uint16),
    ("y", np.uint16),
    ("p", np.int8)
])

events_per_frame = 5000

with open(FILE, "rb") as f:

    while True:

        chunk = np.fromfile(f, dtype=dtype, count=events_per_frame)

        if len(chunk) == 0:
            break

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        x = chunk["x"]
        y = chunk["y"]
        p = chunk["p"]

        frame[y[p > 0], x[p > 0]] = (255,255,255)
        frame[y[p <= 0], x[p <= 0]] = (0,0,255)

        cv2.imshow("Event Viewer", frame)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()