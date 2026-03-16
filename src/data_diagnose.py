import numpy as np

f = open("test/moorea_2019-01-30_000_td_671500000_731500000_td.dat", "rb")

data = np.fromfile(f, dtype=np.uint32, count=20)

print(data)