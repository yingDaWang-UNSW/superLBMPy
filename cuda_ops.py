# cuda_ops.py
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

def allocate_on_gpu(block):
    # Convert the block to a CUDA array
    block_gpu = cuda.to_device(block)
    return block_gpu

def add_one_int8(block_gpu, block_size):
    # Define a CUDA kernel to multiply a block by 2
    mod = SourceModule("""
    __global__ void add_one_int8(char *block, int size)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < size)
            block[i] += 1;
    }
    """)

    add_one_int8 = mod.get_function("add_one_int8")

    # Launch the kernel
    threads_per_block = 256
    blocks_per_grid = (block_size + threads_per_block - 1) // threads_per_block
    add_one_int8(block_gpu, np.int64(block_size), block=(threads_per_block,1,1), grid=(blocks_per_grid,1))

    return block_gpu


def retrieve_from_gpu(block_gpu, shape, dtype):
    # Copy the block back to host memory
    block = cuda.from_device(block_gpu, shape, dtype)
    return block

