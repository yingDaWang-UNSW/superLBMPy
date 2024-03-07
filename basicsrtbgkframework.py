import numpy as np
import matplotlib.pyplot as plt
import tifffile
import time
# Lattice parameters
nx, ny, nz = 150, 150, 3  # Dimensions of the lattice
tau = 0.6  # Relaxation time
rho_inlet = 1.02  # Inlet pressure in terms of density (rho = pressure / c_s^2, where c_s is the speed of sound)
rho_outlet = 1.00  # Outlet pressure in terms of density
bodyForce=[1e-5,0,0]
nu = 1/3 * (tau - 0.5)  # Kinematic viscosity, related to tau
timesteps = 10000  # Total number of timesteps to simulate

# LBM parameters
q = 19  # D3Q19 model
csq = 1/3  # Speed of sound squared

# Direction vectors and weights for D3Q19
ex = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1])
ey = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 1, -1, 1, -1, 0, 0, 0, 0])
ez = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1])
w = np.array([1/3] + [1/18]*6 + [1/36]*12)
# Correct bounce-back mapping for D3Q19
opposite_directions = [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15]

# Initialize the distribution function
f = np.ones((nx, ny, nz, q)) * rho_outlet / q  # Initialize with outlet density
# feq = np.empty_like(f)

# Binary porous media mask
obstacle = tifffile.imread('geopack.tiff')
obstacle = obstacle[0:nx,0:ny,0:nz].astype(bool)
rho = np.sum(f, axis=3)

# # idenitfy neighbours and bouncebacks
# globInds=np.reshape(np.arange(nx*ny*nz),[nx,ny,nz])


# neighbours=f
# for i in range(q):
#     neighbours[:, :, :, i] = np.roll(globInds[:, :, :], ex[i], axis=0)
#     neighbours[:, :, :, i] = np.roll(globInds[:, :, :], ey[i], axis=1)
#     neighbours[:, :, :, i] = np.roll(globInds[:, :, :], ez[i], axis=2)
# # Apply bounce-back for the neighbours
# for i in range(q):
#     neighbours[obstacle, i] = neighbours[obstacle, opposite_directions[i]]

# activeInds=globInds[~obstacle]

# f_opt=np.reshape(f,[-1,q])
# neighbours=np.reshape(neighbours,[-1,q])

# f_opt=f[activeInds,:]
# neighbours=neighbours[activeInds,:]

# Turn on interactive mode
# plt.ion()

# # Initial plot setup
# fig, ax = plt.subplots()
# image = ax.imshow(rho[:, :, 1] - 1, cmap='jet', vmin=0, vmax=rho_inlet-1)  # Adjust the colormap as needed
# ax.set_title('Density Variation')

# fig2, ax2 = plt.subplots()
# image2 = ax2.imshow(rho[:, :, 1] - 1, cmap='jet', vmin=0, vmax=1e-5)  # Adjust the colormap as needed
# ax2.set_title('vMag')

# Main loop
start=time.time()
for t in range(timesteps):
    # Compute macroscopic variables (density, velocity)
    rho = np.sum(f, axis=3)
    ux = np.sum(f * ex, axis=3) / rho
    uy = np.sum(f * ey, axis=3) / rho
    uz = np.sum(f * ez, axis=3) / rho

    # Compute, collide, and stream equilibrium distribution function (for the rest of the domain)
    u_sq = ux**2 + uy**2 + uz**2
    for i in range(q):
        vu = ux*ex[i] + uy*ey[i] + uz*ez[i]
        bf = bodyForce[0]*w[0]*ex[i] + bodyForce[1]*w[1]*ey[i] + bodyForce[2]*w[2]*ez[i]
        f[:, :, :, i] += -(1.0/tau) * (f[:, :, :, i] - w[i] * rho * (1 + 3*vu + 9/2*vu**2 - 3/2*u_sq) + bf)

        f[:, :, :, i] = np.roll(f[:, :, :, i], ex[i], axis=0)
        f[:, :, :, i] = np.roll(f[:, :, :, i], ey[i], axis=1)
        f[:, :, :, i] = np.roll(f[:, :, :, i], ez[i], axis=2)

    # Apply bounce-back for the obstacle
    for i in range(q):
        f[obstacle, i] = f[obstacle, opposite_directions[i]]


    if np.mod(t,10)==0:
        current=time.time()-start
        MLUPS=np.sum(~obstacle)*(t+1)/1000000/current
        print(f't:, {t}, time: {current}s, MLUPS: {MLUPS}')
    # Optional: visualization or data analysis can be added here
    # rho[obstacle]=1
    # image.set_data(rho[:, :, 1] - 1)  # Update the data shown in the plot
    # ax.set_title(f'Density Variation {time}')
    # u_sq[obstacle]=0
    # image2.set_data(u_sq[:, :, 1])  # Update the data shown in the plot
    # ax2.set_title(f'vMag {time}')

    # plt.draw()
    # plt.pause(0.001)  # Pause to update the plot, adjust the time as needed for your simulation speed
    # print(f'{u_sq[:,:,5]-1}')
# Final visualization or analysis
