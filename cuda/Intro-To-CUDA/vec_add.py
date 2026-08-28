import numpy as np
from numba import cuda
import cupy as cp

## Defines a CUDA kernel to perform C = A + B vector addition
@cuda.jit
def vec_add(A, B, C):
  work_index = cuda.grid(1)
  C[work_index] = A[work_index] + B[work_index]

vector_size = 2**24 + 11

device = cp.cuda.Device()
## Create device arrays of uniform random float32 values as input, and an array of zeros 
## as the result vector
a = cp.random.uniform(-1, 1, vector_size)
b = cp.random.uniform(-1, 1, vector_size)
c = cp.zeros(vector_size)

block_size = 256
grid_size = int(np.ceil(vector_size/block_size))
vec_add[grid_size, block_size](a, b, c)

device.synchronize()

# Copy all 3 arrays to the CPU as ndarrays
a_np = cp.asnumpy(a)
b_np= cp.asnumpy(b)
c_np = cp.asnumpy(c)

# Perofrm the copy on the CPU to verify the answer
expected = a_np + b_np

#  Test that the answer is correct, within floating point epsilon
np.testing.assert_array_almost_equal(c_np, expected)

print("Test succeeded")
