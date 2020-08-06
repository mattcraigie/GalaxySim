import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files (x86)\\FFMpeg\\bin\\ffmpeg.exe'

np.random.seed(4080)

# Setup
n_dimensions = 2
n_particles = 5
n_frames = 100
frame_duration = 100
v_max = 0.01  # maximum drift velocity (distance per timestep)

# Set initial positions and velocities
position = 1 - 2 * np.random.random((n_dimensions, n_particles))
velocity = v_max * (1 - 2 * np.random.random((n_dimensions, n_particles)))


def apply_boundary(p):
    """Defines the boundary conditions - we want a periodic boundary
    todo: un-hard code the periodic number
    """
    # i is x, y, z while j is particle 1, particle 2...
    for i in range(n_dimensions):
        for j in range(n_particles):
            if p[i, j] > 1:
                p[i, j] -= 2
            elif p[i, j] < -1:
                p[i, j] += 2
    return p


def apply_force(p, v):
    """Defines the interactions - we want a softening length to avoid crazy big forces. It should be ~ the mean
    separation length of two particles of mass M"""


    for j in range(n_particles):
        # particle-particle interaction
        pass
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
    position += velocity  # Increment positions according to their velocites
    position = apply_boundary(position)  # Apply boundary conditions
    points.set_data(position[0, :], position[1, :])  # Show 2D projection of first 2 position coordinates
    return points,



ani = FuncAnimation(fig, update, interval=300)


f = r"sim.mp4"
writervideo = FFMpegWriter(fps=60)
ani.save(f, writer=writervideo)

plt.show()




