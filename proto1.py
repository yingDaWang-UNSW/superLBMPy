import args
from mpi4py import MPI
import pycuda.driver as cuda
import tifffile
from sys import stdout
import numpy as np
import os
from glob import glob
import time
import datetime
import pdb
from matplotlib import pyplot as plt
from sys import stdout
from cuda_ops import allocate_on_gpu, add_one_int8, retrieve_from_gpu
from decomp_domain import split_into_blocks, recombine_blocks, get_neighbor_ranks, init_send_recv_buffers, memoryOptimizedAADomain
#def main():
import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2int(v):
    if v=='M':
        return v
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError('int value expected.')
    return v
    
def str2float(v):
    if v=='M':
        return v
    try:
        v = float(v)
    except:
        raise argparse.ArgumentTypeError('float value expected.')
    return v
def args():
    # TODO modularise this into concentricGAN noise->clean->SR->seg->vels
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpuIDs', dest='gpuIDs', type=str, default='1,2,3', help='IDs for the GPUs. Empty for CPU. Nospaces')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='/home/user/Insync/sourceCodes/superLBMPy/geopack.tiff', help='dataset path - include last slash')

    parser.add_argument('--nx', dest='nx', type=str2int, default=2, help='# 3D images in batch')
    parser.add_argument('--ny', dest='ny', type=str2int, default=2, help='# 3D images in batch')
    parser.add_argument('--nz', dest='nz', type=str2int, default=2, help='# 3D images in batch')


    parser.add_argument('--lx', dest='lx', type=str2int, default=75, help='# 3D images in batch')
    parser.add_argument('--ly', dest='ly', type=str2int, default=75, help='# 3D images in batch')
    parser.add_argument('--lz', dest='lz', type=str2int, default=75, help='# 3D images in batch')
    
    
    args = parser.parse_args()

    return args

args=args() # args is global

gpuList=args.gpuIDs
args.numGPUs = len(gpuList.split(','))
if args.numGPUs<=4:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuList
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

lx=args.lx
ly=args.ly
lz=args.lz

nx=args.nx
ny=args.ny
nz=args.nz

tau=0.7

if rank == 0:
    print("MPI Initialised")
    print("Welcome to SuperLBMPy. This is the prototype 3D Single Phase SRT/MRT Module")
    # print('WARNING: SuperLBMPy is designed to ONLY run on CUDA GPUs - inherent LBM performance on CPUs is pathetic anyway.')
comm.Barrier()

print(f"CPU core {rank} online")

if rank == 0:
    print(f"Reading Input Domain {args.dataset_dir}")
    # Read the TIFF file
    image = tifffile.imread(args.dataset_dir)
    # Split the image into blocks
    print(f"Decomposing Domain...")
    blocks, positions = split_into_blocks(image, lx, ly, lz)
else:
    blocks = positions = None

if rank==0: print(f"Scattering domain to ranks...");
# Scatter the blocks to all processes
block = comm.scatter(blocks, root=0)
position = comm.scatter(positions, root=0)

# Gather the blocks back to rank 0
gathered_blocks = comm.gather(block, root=0)
gathered_positions = comm.gather(position, root=0)
if rank == 0:
    # Combine the gathered blocks back into an image
    gathered_image = recombine_blocks(gathered_blocks, gathered_positions, image.shape)
    # Check if the gathered image is the same as the original image
    print(f'Input Domain: {image.shape}, Recombined Domain: {gathered_image.shape}. Integrity Check Status:')    
    print(np.array_equal(image, gathered_image))

# prepare the domain for simulation 
if rank==0: print(f"Identifying rank neighbours");
# identify the neighbouring ranks with periodicity
neighbor_ranks = get_neighbor_ranks([nx,ny,nz], comm)
#print(f"Rank {comm.Get_rank()} neighbors:") 
#print(f"{neighbor_ranks} ")
if rank==0: print(f"Indexing active communication voxels")
# set up the communication buffers using the neighbouring rank info and block geometry
sendCounts, sendInds, sendBuffers, sendData, recvCounts, recvInds, recvBuffers, recvData = init_send_recv_buffers(block,comm,neighbor_ranks)
# set up the memory efficient communication buffers using the neighbouring rank info and block geometry
neighborList, sendIndsCompact, recvIndsCompact, compactIndMap = memoryOptimizedAADomain(block,comm,neighbor_ranks,sendInds,recvInds,)

# initialise all communications buffers in GPU devices

#print(f"sendCounts: ")
#print(f"{sendCounts}")
##print(f"sendInds: ")
##print(f"{sendInds}")
#print(f"recvCounts: ")
#print(f"{recvCounts}")
#print(f"recvInds: ")
#print(f"{recvInds}")
# create the memory efficient layout and compute mean porosity
# neighbourlist, fq, pressure, velocity, 

# create cartesian vx,vy,vz,density in cpu - these can just be np arrays

# init cuda and the gpus
cuda.init()
comm.Barrier()
if rank == 0:
   print("CUDA Initialised")
comm.Barrier()
# Assign a GPU to this process
gpu_id = rank % cuda.Device.count()
gpu = cuda.Device(gpu_id)
print(f"GPU Device: {gpu.name()}, {gpu.pci_bus_id()} online - linked to rank {rank}")

## Make this GPU the current one for this process
context = gpu.make_context()

# Now you can run your CUDA code...
# pass fields to cuda

# alllocate the F field - 


# Allocate the block on the GPU
block_gpu = allocate_on_gpu(block)
# add 1 to the block using CUDA
block_gpu = add_one_int8(block_gpu, block.size)
# Retrieve the block from the GPU
block = retrieve_from_gpu(block_gpu, block.shape, block.dtype)


# Gather the blocks back to rank 0
gathered_blocks = comm.gather(block, root=0)
gathered_positions = comm.gather(position, root=0)

if rank == 0:
   # Recombine the gathered blocks back into an image
   gathered_image = recombine_blocks(gathered_blocks, gathered_positions, image.shape)

   # Check if the gathered image is the same as the original image multiplied by 2
   print(np.array_equal(image +1, gathered_image))

# Clean up
context.pop()
    
    
#if __name__ == "__main__":
#    main()

