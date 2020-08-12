import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.fft import fftn, ifftn
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
n_frames = 10
frame_duration = 100


# physical quantities - everything SI
particle_mass = 10**11 * 1.98e30            # kg
G = 6.67e-11                                # m^3 kg^-1 s^-2
v_max = 1000 * 1000                         # m/s
side_length = 1e5 * 3.086e16                # in units of m
radius_buffer = 0.1 * side_length           # in units of m
dt = 3.154e+7 * 1e5                         # in units of seconds

# physical quantities - everything SI
# particle_mass = 1            # kg
# G = 1                           # m^3 kg^-1 s^-2
# v_max = 0.05                         # m/s
# side_length = 1                # in units of m
# radius_buffer = 0.1 * side_length           # in units of m
# dt = 0.01                         # in units of seconds



# Set initial positions and velocities
position = side_length * (0.5 - np.random.random((n_dimensions, n_particles)))
velocity = v_max * (0.5 - np.random.random((n_dimensions, n_particles)))


def apply_boundary(p):
    """Defines the boundary conditions - we want a periodic boundary"""
    p[p > side_length/2] -= side_length
    p[p < -side_length/2] += side_length
    return p


def apply_force_pp(p, v):
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


def apply_force_pm(p, v):
    cell_num = 50
    rho = ngp(p, cell_num)
    rho_ft = fftn(rho)

    x = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1]
    y = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1]
    z = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1]

    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    k_x, k_y, k_z = 2 * np.pi * grid_x / side_length, 2 * np.pi * grid_y / side_length, 2 * np.pi * grid_z / side_length,
    k_squared = k_x**2 + k_y**2 + k_z**2

    phi_ft = -4 * np.pi * G * rho_ft / k_squared

    phi = ifftn(phi_ft)
    phi = np.abs(phi)

    g = np.gradient(phi)

    # to_plot = np.sum(phi, axis=1)
    # plt.close()
    # plt.imshow(to_plot)
    # plt.savefig('yep.png')
    # exit()

    x = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1]
    for j in range(n_particles):
        coords = []
        for i in range(n_dimensions):
            coords.append(np.argmin(p[i, j] - x))
        for i in range(n_dimensions):
            v[i, j] = g[i][coords[0]][coords[1]][coords[2]]
    return p, v


def ngp(positions, cell_num):
    """Gets a mass grid of the particles for a simple ngp method"""
    x = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1] * side_length
    mass_grid = np.zeros([cell_num, cell_num, cell_num])
    for j in range(n_particles):
        grid_position = np.zeros(n_dimensions, dtype=int)
        for i in range(n_dimensions):
            grid_position[i] = int(np.argmin(np.abs(positions[i, j] - x)))
        mass_grid[grid_position[0], grid_position[1], grid_position[2]] += 1  # not n-dimensional
    mass_grid = mass_grid * particle_mass

    return mass_grid


def greens_function(cell_num):
    x = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1] * side_length
    k = 2 * np.pi * x / side_length
    G = np.zeros([cell_num, cell_num, cell_num])
    for l in range(cell_num):
        for m in range(cell_num):
            for n in range(cell_num):
                G[l][m][n] = -3 / 8 * (np.sin(k[l] / 2)**2 +
                                       np.sin(k[m] / 2)**2 +
                                       np.sin(k[n] / 2)**2)**-1
    return G



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
    position, velocity = apply_force_pp(position, velocity)
    points.set_data(position[0, :], position[1, :])  # Show 2D projection of first 2 position coordinates
    return points,


ani = FuncAnimation(fig, update, frames=n_frames, interval=frame_duration)
f = r"sim.mp4"
writervideo = FFMpegWriter(fps=60)
ani.save(f, writer=writervideo)


