import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files (x86)\\FFMpeg\\bin\\ffmpeg.exe'

np.random.seed(42)

# todo: physical units
# todo: introduce PM approach
# todo: force wrapping
# todo: gaussian (or otherwise) mass distribution for each particle
# todo: PM to arbitrary dimension

km_to_pc = 3.240e-14

# Sim setup
n_dimensions = 3
n_particles = 10
n_frames = 1000
frame_duration = 100


# physical quantities - everything SI
particle_mass = 10**11 * 1.98e30            # kg
G = 6.67e-11                                # m^3 kg^-1 s^-2
v_max = 1000 * 1000                         # m/s
side_length = 1e5 * 3.086e16                # in units of m
radius_buffer = 0.1 * side_length           # in units of m
dt = 3.154e+7 * 1e5                         # in units of seconds


# Set initial positions and velocities
position = side_length * (0.5 - np.random.random((n_dimensions, n_particles)))
velocity = v_max * (0.5 - np.random.random((n_dimensions, n_particles)))


def apply_boundary(p):
    """Defines the boundary conditions - we want a periodic boundary"""
    p[p > side_length/2] -= side_length
    p[p < -side_length/2] += side_length
    return p


def apply_force_particle(p, v):
    """Returns the velocity in km/s (weird)

    Defines the interactions - we want a softening length to avoid crazy big forces. It should be ~ the mean
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
        # this might also need some 4s and pis etc.
        v[:, m] += - G * particle_mass * distance_factor * dt
    return p, v


def apply_force_particle_mesh(p, v):
    grid_num = 50
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    z = np.linspace(-1, 1, 50)
    pos_grid = np.meshgrid(x, y, z)
    mass_grid = np.zeros([grid_num] * n_dimensions)
    H = side_length / grid_num

    print(pos_grid)
    print(mass_grid)
    exit()

    for i in range(n_particles):
        diff = p[:, i] - pos_grid
        p[:, i]

    rho_grid = mass_grid / H**3

def pm_mass_assignment_func(m, g, H):




    return m * W

def ngp(d, H):
    if d <= H/2:
        return 1
    else:
        return 0





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
    position, velocity = apply_force_particle_mesh(position, velocity)
    points.set_data(position[0, :], position[1, :])  # Show 2D projection of first 2 position coordinates
    return points,


ani = FuncAnimation(fig, update, frames=n_frames)
f = r"sim.mp4"
writervideo = FFMpegWriter(fps=60)
ani.save(f, writer=writervideo)


