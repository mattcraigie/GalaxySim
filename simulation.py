import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files (x86)\\FFMpeg\\bin\\ffmpeg.exe'

np.random.seed(4080)

# possible extensions: gaussian (or otherwise) mass distribution for each particle


# Sim setup
n_dimensions = 3
n_particles = 10
n_frames = 300
frame_duration = 100
dt = 1

# physical quantities
particle_mass = 1
radius_buffer = 0.1
G = 1e-4
v_max = 0.03  # starting maximum velocity for the particles
side_length = 1


# Set initial positions and velocities
position = 0.5 * side_length - side_length * np.random.random((n_dimensions, n_particles))
velocity = v_max * (0.5 * side_length - side_length * np.random.random((n_dimensions, n_particles)))


def apply_boundary(p):
    """Defines the boundary conditions - we want a periodic boundary
    """
    p[p > side_length/2] -= side_length
    p[p < -side_length/2] += side_length

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


# Set the axes on which the points will be shown
plt.ion() # Set interactive mode on
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-side_length*0.5, side_length*0.5)
ax.set_ylim(-side_length*0.5, side_length*0.5)


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


