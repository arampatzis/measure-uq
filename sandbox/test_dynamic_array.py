#!/usr/bin/env python3
import pickle

from measure_uq.utilities import DynamicArray

x = DynamicArray(shape=2, index_expansion=True)

x[0] = 0

for i in range(1, 5):
    x[i] = i

x[2:10] = -1
x[15] = -2
x[17] = -3

print(x)
print(x.data)
print(len(x))
print(x.capacity)
print(len(x._data))

with open("tmp.pickle", "wb") as f:
    pickle.dump(x, f)

with open("tmp.pickle", "rb") as f:
    y = pickle.load(f)

print(y)
