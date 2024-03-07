from mpi4py import MPI
import numpy as np
import pdb

    
def split_into_blocks(image, nx, ny, nz):
    blocks = []
    positions = []
    for i in range(0, image.shape[0], nx):
        for j in range(0, image.shape[1], ny):
            for k in range(0, image.shape[2], nz):
                block = image[i:i+nx, j:j+ny, k:k+nz]

                block=np.pad(block,1,'edge')
                blocks.append(block)
                positions.append((i, j, k))
    return blocks, positions
    
def recombine_blocks(blocks, positions, shape):
    image = np.empty(shape, blocks[0].dtype)
    for block, (i, j, k) in zip(blocks, positions):
        image[i:i+block.shape[0]-2, j:j+block.shape[1]-2, k:k+block.shape[2]-2] = block[1:-1,1:-1,1:-1]
    return image
    
def get_neighbor_ranks(dims, comm):
    # Setup Cartesian grid dimensions
    periodic = [True, True, True] # Grid is periodic in all directions
    reorder = False # Allow reordering of ranks

    # Create Cartesian communicator
    cart_comm = comm.Create_cart(dims, periods=periodic, reorder=reorder)

    # Get rank and coordinates of this process
    my_rank = cart_comm.Get_rank()
    rank = comm.Get_rank()
    my_coords = cart_comm.Get_coords(my_rank)

    neighbor_ranks = np.empty((3, 3, 3), dtype=object)  # Create an empty 3x3x3 NumPy array
    directions = [-1, 0, 1] # Three possible displacements in each dimension

    for dx in directions:
        for dy in directions:
            for dz in directions:
                coords = [(my_coords[0] + dx) % dims[0], 
                          (my_coords[1] + dy) % dims[1], 
                          (my_coords[2] + dz) % dims[2]]
                neighbour_rank = cart_comm.Get_cart_rank(coords)
                neighbor_ranks[dx+1, dy+1, dz+1] = (neighbour_rank, coords)
    return neighbor_ranks
    
def init_send_recv_buffers(block,comm,neighbor_ranks):
    # count up the sends 
#    blocktags = ['dir_'+tag for tag in ['x', 'y', 'z', 'X', 'Y', 'Z', 'xy', 'yz', 'xz', 'Xy', 'Yz', 'xZ', 'xY', 'yZ', 'Xz', 'XY', 'YZ', 'XZ']]
    rank = comm.Get_rank()
    sendCounts = np.zeros((3,3,3), dtype=np.int32)
    sendInds = np.empty((3, 3, 3), dtype=object)
    sendBuffers = np.empty((3, 3, 3), dtype=object)
    sendData = np.empty((3, 3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                sendInds[i, j, k] = []
                sendBuffers[i, j, k] = []
                sendData[i, j, k] = []
    Nx = block.shape[0]
    Ny = block.shape[1]
    Nz = block.shape[2]
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                n = k*Nx*Ny + j*Nx + i
                # Check the phase ID
                if block[i, j, k] == 0:
                    # Counts for the six faces
                    if i == 1: sendCounts[0,1,1] += 1; sendInds[0,1,1].append(n); sendBuffers[0,1,1].append(0); sendData[0,1,1].append(0.)
                    if j == 1: sendCounts[1,0,1] += 1; sendInds[1,0,1].append(n); sendBuffers[1,0,1].append(0); sendData[1,0,1].append(0.)
                    if k == 1: sendCounts[1,1,0] += 1; sendInds[1,1,0].append(n); sendBuffers[1,1,0].append(0); sendData[1,1,0].append(0.)
                    if i == Nx-2: sendCounts[2,1,1] += 1; sendInds[2,1,1].append(n); sendBuffers[2,1,1].append(0); sendData[2,1,1].append(0.)
                    if j == Ny-2: sendCounts[1,2,1] += 1; sendInds[1,2,1].append(n); sendBuffers[1,2,1].append(0); sendData[1,2,1].append(0.)
                    if k == Nz-2: sendCounts[1,1,2] += 1; sendInds[1,1,2].append(n); sendBuffers[1,1,2].append(0); sendData[1,1,2].append(0.)
                    # Counts for the twelve edges
                    if i == 1 and j == 1: sendCounts[0,0,1] += 1; sendInds[0,0,1].append(n); sendBuffers[0,1,1].append(0); sendData[0,1,1].append(0.)
                    if i == 1 and j == Ny-2: sendCounts[0,2,1] += 1; sendInds[0,2,1].append(n); sendBuffers[0,2,1].append(0); sendData[0,2,1].append(0.)
                    if i == Nx-2 and j == 1: sendCounts[2,0,1] += 1; sendInds[2,0,1].append(n); sendBuffers[2,0,1].append(0); sendData[2,0,1].append(0.)
                    if i == Nx-2 and j == Ny-2: sendCounts[2,2,1] += 1; sendInds[2,2,1].append(n); sendBuffers[2,2,1].append(0); sendData[2,2,1].append(0.)

                    if i == 1 and k == 1: sendCounts[0,1,0] += 1; sendInds[0,1,0].append(n); sendBuffers[0,1,0].append(0); sendData[0,1,0].append(0.)
                    if i == 1 and k == Nz-2: sendCounts[0,1,2] += 1; sendInds[0,1,2].append(n); sendBuffers[0,1,2].append(0); sendData[0,1,2].append(0.)
                    if i == Nx-2 and k == 1: sendCounts[2,1,0] += 1; sendInds[2,1,0].append(n); sendBuffers[2,1,0].append(0); sendData[2,1,0].append(0.)
                    if i == Nx-2 and k == Nz-2: sendCounts[2,1,2] += 1; sendInds[2,1,2].append(n); sendBuffers[2,1,2].append(0); sendData[2,1,2].append(0.)

                    if j == 1 and k == 1: sendCounts[1,0,0] += 1; sendInds[1,0,0].append(n); sendBuffers[1,0,0].append(0); sendData[1,0,0].append(0.)
                    if j == 1 and k == Nz-2: sendCounts[1,0,2] += 1; sendInds[1,0,2].append(n); sendBuffers[1,0,2].append(0); sendData[1,0,2].append(0.)
                    if j == Ny-2 and k == 1: sendCounts[1,2,0] += 1; sendInds[1,2,0].append(n); sendBuffers[1,2,0].append(0); sendData[1,2,0].append(0.)
                    if j == Ny-2 and k == Nz-2: sendCounts[1,2,2] += 1; sendInds[1,2,2].append(n); sendBuffers[1,2,2].append(0); sendData[1,2,2].append(0.)

                    
    # send the sends to get the recvs

    recvCounts = np.empty((3,3,3), dtype=np.int32)
    recvInds = np.empty((3, 3, 3), dtype=object)
    recvBuffers = np.empty((3, 3, 3), dtype=object)
    recvData = np.empty((3, 3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                recvInds[i, j, k] = []
                recvBuffers[i, j, k] = []
                recvData[i, j, k] = []
    req1 = []
    req2 = []
    sendtag = 69
    recvtag = 69
    directions = [0,1,2] # Three possible displacements in each dimension
    
#    C =[
#        [ 1, 0, 0], 
#        [-1, 0, 0], 
#        [ 0, 1, 0], 
#        [ 0,-1, 0], 
#        [ 0, 0, 1], 
#        [ 0, 0,-1], 
#        [ 1, 1, 0], 
#        [-1,-1, 0], 
#        [ 1,-1, 0], 
#        [-1, 1, 0], 
#        [ 1, 0, 1], 
#        [-1, 0,-1],
#        [ 1, 0,-1], 
#        [-1, 0, 1], 
#        [ 0, 1, 1], 
#        [ 0,-1,-1], 
#        [ 0, 1,-1], 
#        [ 0,-1, 1]]
#    dx=0;dy=1;dz=1
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=2;dy=1;dz=1
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=1;dy=0;dz=1
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=1;dy=2;dz=1
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=1;dy=1;dz=0
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=1;dy=1;dz=2
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=0;dy=0;dz=1 
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=2;dy=2;dz=1
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=0;dy=2;dz=1
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=2;dy=0;dz=1
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    
#    dx=0;dy=1;dz=0
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=2;dy=1;dz=2
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=0;dy=1;dz=2
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=2;dy=1;dz=0
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=1;dy=0;dz=0
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=1;dy=2;dz=2
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=1;dy=0;dz=2
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
#    dx=1;dy=2;dz=0
#    sc=np.array([sendCounts[dx,dy,dz]]); rc=np.array([0], dtype=np.int32)
#    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#    req1.append(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#    req2.append(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
#    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
#    sendtag+=1; recvtag+=1
#    
    for dx in directions:
        for dy in directions:
            for dz in directions:
#                if dx!=1 or dy!=1 or dz!=1:
                    sc=np.array([sendCounts[dx,dy,dz]])
                    rc=np.array([0], dtype=np.int32)
#                    print(f'Sending {sc} from rank {rank} to {neighbor_ranks[dx,dy,dz][0]} in the direction {[dx-1,dy-1,dz-1]}')                   
                    # at this rank, at face x,y,z, send data at xyz to the neighbor rank of this face
                    req1=(comm.Isend(sc, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
                    # at this rank, for the opposite face XYZ, receive data from the neighbor rank to the opposite face
                    req2=(comm.Irecv(rc, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
                    MPI.Request.Wait(req1)
                    MPI.Request.Wait(req2)
#                    print(f'Received {rc} from rank {neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0]} to {rank} in the direction {[abs(dx-2)-1,abs(dy-2)-1,abs(dz-2)-1]}')
                    recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]=rc
                    sendtag+=1
                    recvtag+=1
    comm.Barrier()

    # Compute the local sums
    local_send_sum = np.sum(sendCounts)
    local_recv_sum = np.sum(recvCounts)

    # Compute the global sums
    global_send_sum = comm.allreduce(local_send_sum, op=MPI.SUM)
    global_recv_sum = comm.allreduce(local_recv_sum, op=MPI.SUM)
    if rank==0:
        if global_send_sum != global_recv_sum:
            print("The total sum of the arrays over all ranks is not the same.")
            print(f"rank {rank} sendCounts: {sendCounts}")
        else:
            print('Send Receive Counts Confirmed Match')


#    req1 = []
#    req2 = []
#    sendtag = 69
#    recvtag = 69
##    for dx in directions:
##        for dy in directions:
##            for dz in directions:
##                if dx!=1 or dy!=1 or dz!=1:
#    for x in C:
#        dx=x[0]+1
#        dy=x[1]+1
#        dz=x[2]+1
#        sl=np.array([sendInds[dx,dy,dz]])
#        rl=np.zeros([recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]], dtype=np.int64)

##        recvBuffers[abs(dx-2),abs(dy-2),abs(dz-2)]=rl.tolist()
##        recvData[abs(dx-2),abs(dy-2),abs(dz-2)]=(rl*0.0).tolist()
#        
#        if len(sl) > 0 or len(rl) > 0:
#            try:

#                # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
#                req1.append(comm.Isend(sl, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
#                # at this rank, at face xyz, receive data from the opposite face of neighbor rank
#                req2.append(comm.Irecv(rl, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
##                print(f"MPI rank: {rank}")
##                print(f"dx: {dx}, dy: {dy}, dz: {dz}")
##                print(f"sendtag: {sendtag}, recvtag: {recvtag}")
##                print(f"sl: {sl}, rl: {rl}")
#            except Exception as e:
#                print(f"Exception occurred during MPI operation: {e}")
#                print(f"dx: {dx}, dy: {dy}, dz: {dz}")
#                print(f"sendtag: {sendtag}, recvtag: {recvtag}")
#                print(f"sl: {sl}, rl: {rl}")
#            sendtag+=1
#            recvtag+=1
##                        


#    try:
#        MPI.Request.Waitall(req1)
#        MPI.Request.Waitall(req2)
#    except Exception as e:
#        print(f"Exception occurred during MPI Waitall operation: {e}")
#    comm.Barrier()

#    for i in range(3):
#        for j in range(3):
#            for k in range(3):
#                if len(recvInds[i,j,k]) != recvCounts[i,j,k]:
#                    raise ValueError(f"Discrepancy at ({i}, {j}, {k}): Length of recvInds is {len(recvInds[i,j,k])}, while recvCounts is {recvCounts[i,j,k]}")
    req1 = []
    req2 = []
    sendtag = 69
    recvtag = 69
    for dx in directions:
        for dy in directions:
            for dz in directions:
                if dx!=1 or dy!=1 or dz!=1:
                
                    sl=np.array([sendInds[dx,dy,dz]])
                    rl=np.zeros([recvCounts[abs(dx-2),abs(dy-2),abs(dz-2)]],dtype=np.int64)

                    recvBuffers[abs(dx-2),abs(dy-2),abs(dz-2)]=rl.tolist()
                    recvData[abs(dx-2),abs(dy-2),abs(dz-2)]=(rl*0.0).tolist()
                    
                    # at this rank, at face x,y,z, send data at xyz to opposite face of neighbor rank
                    req1=(comm.Isend(sl, dest=neighbor_ranks[dx,dy,dz][0], tag=sendtag))
                    # at this rank, at face xyz, receive data from the opposite face of neighbor rank
                    req2=(comm.Irecv(rl, source=neighbor_ranks[abs(dx-2),abs(dy-2),abs(dz-2)][0], tag=recvtag))
                    MPI.Request.Wait(req1)
                    MPI.Request.Wait(req2)
                    sendtag+=1
                    recvtag+=1
                    
                    rl = rl + (Nx-2)*(abs(dx-2)-1) + (Ny-2)*Nx*(abs(dy-2)-1) + (Nz-2)*Nx*Ny*(abs(dz-2)-1) # make received inds local again
                    recvInds[abs(dx-2),abs(dy-2),abs(dz-2)]=rl.tolist()

    comm.Barrier()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len(recvInds[i,j,k]) != recvCounts[i,j,k]:
                    raise ValueError(f"Discrepancy at ({i}, {j}, {k}): Length of recvInds is {len(recvInds[i,j,k])}, while recvCounts is {recvCounts[i,j,k]}")
    # Compute the local sums
    local_send_sum = np.sum(np.sum(sendInds))
    local_recv_sum = np.sum(np.sum(recvInds))

    # Compute the global sums
    global_send_sum = comm.allreduce(local_send_sum, op=MPI.SUM)
    global_recv_sum = comm.allreduce(local_recv_sum, op=MPI.SUM)
#    if rank ==0:
#        if global_send_sum != global_recv_sum:
#            print(f"The total sum of the arrays over all ranks is not the same. {global_send_sum} vs {global_recv_sum}")

#    #        print(f"sendInds: ")
#    #        print(f"{sendInds}")

#    #        print(f"recvInds: ")
#    #        print(f"{recvInds}")
#        else:
#            print('Send Receive Indexes Confirmed Match')

    return sendCounts, sendInds, sendBuffers, sendData, recvCounts, recvInds, recvBuffers, recvData
    
    
    
def memoryOptimizedAADomain(block,comm,neighbor_ranks,sendInds,recvInds):
    rank = comm.Get_rank()
    if rank==0: print('Indexing send/recv of active voxels and neighbour map')

    return neighborList, sendIndsCompact, recvIndsCompact, compactIndMap


























































