import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files (x86)\\FFMpeg\\bin\\ffmpeg.exe'

np.random.seed(4080)

# possible extensions: gaussian (or otherwise) mass distribution for each particle


# Setup
n_dimensions = 2
n_particles = 5
n_frames = 100
frame_duration = 100
v_max = 0.01  # maximum drift velocity (distance per timestep)
dt = 1

# physical quantities
particle_mass = 1
radius_buffer = 0.1
G = 1e-4

# Set initial positions and velocities
position = 1 - 2 * np.random.random((n_dimensions, n_particles))
velocity = v_max * (1 - 2 * np.random.random((n_dimensions, n_particles)))


def apply_boundary(p):
    """Defines the boundary conditions - we want a periodic boundary
    todo: un-hard code the periodic number
    """
    # i is x, y, z while j is particle 1, particle 2...

    p[p > 1] -= 2
    p[p < -1] += 2

    return p


def apply_force(p, v):
    """Defines the interactions - we want a softening length to avoid crazy big forces. It should be ~ the mean
    separation length of two particles of mass M.

    the change in velocity (i.e. acceleration) particle m is determined by its interaction with n
    """


    for m in range(n_particles):
        # particle-particle interaction
        distance_factor = 0

        for n in range(n_particles):
            if n == m:
                continue
            distance_factor += (p[:, m] - p[:, n]) / (np.linalg.norm(p[:, m] - p[:, n])**n_dimensions + radius_buffer)
        v[:, m] += - G * distance_factor * dt

    return p, v

    # normal denominator plus a small num

# Set the axes on which the points will be shown
plt.ion() # Set interactive mode on
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)


# Create command which will plot the positions of the particles
points, = plt.plot([], [], 'o', markersize=5)


# Define procedure to update positions at each timestep
def update(i):
    global position, velocity
    position += velocity * dt  # Increment positions according to their velocites
    position = apply_boundary(position)  # Apply boundary conditions
    position, velocity = apply_force(position, velocity)
    points.set_data(position[0, :], position[1, :])  # Show 2D projection of first 2 position coordinates
    return points,


ani = FuncAnimation(fig, update, frames=n_frames)


f = r"sim.mp4"
writervideo = FFMpegWriter(fps=60)
ani.save(f, writer=writervideo)

plt.show()




