#!/usr/bin/env python3
import time

import numpy as np

from measure_uq.utilities import SparseDynamicArray

x = SparseDynamicArray(shape=5, dtype=float)

for i in range(0, 1000, 10):
    x[i] = np.random.uniform(0, 1)

print(x)
print(x[1:10])

print(x[-1])
print(x.v[-1])
print(x.i[-1])

print(x(11))

N = 1000000

start_time = time.perf_counter()

x = SparseDynamicArray(shape=5, dtype=float)

for i in range(N):
    x[i] = i

elapsed = time.perf_counter() - start_time
print(f"Execution time: {elapsed:.6f} seconds")


start_time = time.perf_counter()

y = {}

for i in range(N):
    x[i] = i

elapsed = time.perf_counter() - start_time
print(f"Execution time: {elapsed:.6f} seconds")
