import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import tifffile

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
ex = cp.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1])
ey = cp.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 1, -1, 1, -1, 0, 0, 0, 0])
ez = cp.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1])
w = cp.array([1/3] + [1/18]*6 + [1/36]*12)
# Correct bounce-back mapping for D3Q19
opposite_directions = [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15]

# Initialize the distribution function
f = cp.ones((nx, ny, nz, q)) * rho_outlet / q  # Initialize with outlet density
feq = cp.empty_like(f)

# Binary porous media mask
obstacle = tifffile.imread('geopack.tiff')
obstacle=obstacle[0:nx,0:ny,0:nz].astype(bool)
obstacle = cp.array(obstacle)
rho = cp.sum(f, axis=3)

# Turn on interactive mode
plt.ion()

# Initial plot setup
fig, ax = plt.subplots()
image = ax.imshow(rho[:, :, 1] - 1, cmap='viridis', vmin=0, vmax=rho_inlet-1)  # Adjust the colormap as needed
ax.set_title('Density Variation')

fig2, ax2 = plt.subplots()
image2 = ax2.imshow(rho[:, :, 1] - 1, cmap='viridis', vmin=0, vmax=1e-5)  # Adjust the colormap as needed
ax2.set_title('vMag')

# Main loop
for time in range(timesteps):
    # Compute macroscopic variables (density, velocity)
    rho = cp.sum(f, axis=3)
    ux = cp.sum(f * ex, axis=3) / rho
    uy = cp.sum(f * ey, axis=3) / rho
    uz = cp.sum(f * ez, axis=3) / rho

    # Compute equilibrium distribution function (for the rest of the domain)
    u_sq = ux**2 + uy**2 + uz**2
    for i in range(q):
        vu = ux*ex[i] + uy*ey[i] + uz*ez[i]
        bf = bodyForce[0]*w[0]*ex[i] + bodyForce[1]*w[1]*ey[i] + bodyForce[2]*w[2]*ez[i]
        feq[:, :, :, i] = w[i] * rho * (1 + 3*vu + 9/2*vu**2 - 3/2*u_sq) + bf

    # Collision step
    f += -(1.0/tau) * (f - feq)

    # Streaming step
    for i in range(q):
        f[:, :, :, i] = cp.roll(f[:, :, :, i], ex[i], axis=0)
        f[:, :, :, i] = cp.roll(f[:, :, :, i], ey[i], axis=1)
        f[:, :, :, i] = cp.roll(f[:, :, :, i], ez[i], axis=2)

    # Apply boundary conditions
    # Bounce-back for the obstacle
    # Apply bounce-back for the obstacle
    for i in range(q):
        f[obstacle, i] = f[obstacle, opposite_directions[i]]

    # Optional: visualization or data analysis can be added here
    rho[obstacle]=1
    image.set_data(rho[:, :, 1] - 1)  # Update the data shown in the plot
    ax.set_title(f'Density Variation {time}')
    u_sq[obstacle]=0
    image2.set_data(u_sq[:, :, 1])  # Update the data shown in the plot
    ax2.set_title(f'vMag {time}')

    plt.draw()
    plt.pause(0.001)  # Pause to update the plot, adjust the time as needed for your simulation speed
    # print(f'{u_sq[:,:,5]-1}')
# Final visualization or analysis
