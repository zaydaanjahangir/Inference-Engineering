import numpy as np
import time
from numba import cuda
import cupy as cp

@cuda.jit
def multiply_by_2(arr):
  i = cuda.grid(1)
  if i < arr.size:
    arr[i] = arr[i] * 2

@cuda.jit
def multiply_by_3(arr):
  i = cuda.grid(1)
  if i < arr.size:
    arr[i] = arr[i] * 3

@cuda.jit
def multiply_by_6(arr):
  i = cuda.grid(1)
  if i < arr.size:
    arr[i] = arr[i] * 6

vector_size = 10_000_000
block_size = 256
grid_size = int(np.ceil(vector_size/block_size))

data = np.random.rand(vector_size).astype(np.float32)
d_data = cuda.to_device(data)

# unfused: two kernel launches, two round trips through memory
cuda.synchronize()
start = time.time()
multiply_by_2[grid_size, block_size](d_data)
multiply_by_3[grid_size, block_size](d_data)
cuda.synchronize()
print("Unfused:", time.time() - start)

# fused: one kernel launch, one round trip
d_data2 = cuda.to_device(data)
cuda.synchronize()
start = time.time()
multiply_by_6[grid_size, block_size](d_data2)
cuda.synchronize()
print("Fused:", time.time() - start)

# Results
# Unfused: 0.21319222450256348
# Fused:   0.054671287536621094
