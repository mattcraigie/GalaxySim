import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.fft import fftn, ifftn, rfftn, irfftn
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
n_particles = 500
n_frames = 1000
frame_duration = 100


# physical quantities - everything SI
# particle_mass = 10**11 * 1.98e30            # kg
# G = 5*6.67e-11                                # m^3 kg^-1 s^-2
# v_max = 1000 * 1000                         # m/s
# side_length = 1e5 * 3.086e16                # in units of m
# radius_buffer = 0.1 * side_length           # in units of m
# dt = 3.154e+7 * 1e5                         # in units of seconds

# de-uniting
particle_mass = 1            # kg
G = 1                           # m^3 kg^-1 s^-2
v_max = 0.05                         # m/s
side_length = 1                # in units of m
radius_buffer = 0.1 * side_length           # in units of m
dt = 0.005                         # in units of seconds


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
    particle_coords = get_particle_coords(p, cell_num)
    rho = ngp(particle_coords, cell_num)
    rho_ft = rfftn(rho, axes=(0,1,2))

    x = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1] * side_length
    pos_grid = np.array(np.meshgrid(x, x, x))
    k_grid = 2 * np.pi * pos_grid / side_length
    k_norm = np.linalg.norm(k_grid, axis=0)

    # halving bc rfft is funny
    half = int(cell_num / 2 + 1)
    k_grid = k_grid[:, :, :, :half]
    k_norm = k_norm[:, :, :half]

    a_ft_grid = 4 * 1j * np.pi * G * rho_ft * k_grid / k_norm**2
    a_grid = irfftn(a_ft_grid, axes=(1, 2, 3))

    # print(a_grid[:, 0, 0])
    # # a_mag = np.linalg.nor(a_grid, axis=0)
    # to_plot = np.sum(a_grid[1], axis=2)
    #
    # plt.close()
    # plt.imshow(to_plot)
    # plt.savefig('yvalue.png')
    #
    # exit()

    # inside the range, subtract the mesh force and add the distance force
    # outside the range, do nothing

    for j in range(n_particles):
        j_coords = particle_coords[:, j]
        accel = a_grid[:, particle_coords[0, j], particle_coords[1, j], particle_coords[2, j]]
        
        v[:, j] += accel * dt
    return p, v


def get_particle_coords(p, cell_num):
    x = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1] * side_length
    coords = np.zeros([n_dimensions, n_particles], dtype=int)
    for j in range(n_particles):
        c = []
        for i in range(n_dimensions):
            c.append(int(np.argmin(np.abs(p[i, j] - x))))
        coords[:, j] = np.array(c, dtype=int)
    return coords


def ngp(particle_coords, cell_num):
    """Gets a mass grid of the particles for a simple ngp method"""
    mass_grid = np.zeros([cell_num, cell_num, cell_num])
    for j in range(n_particles):
        grid_position = tuple(particle_coords[:, j])
        mass_grid[grid_position] += 1
    mass_grid = mass_grid * particle_mass
    return mass_grid


def correlation_function_ls(p):
    n_bins = 20
    random_catalog_factor = 10

    n_data = np.shape(p[1])
    n_random_catalog = random_catalog_factor * n_data

    random = create_random_catalog(n_dimensions, n_random_catalog)
    DD = two_point_count(catalog1=p, catalog2=p, n_bins=n_bins)
    DR = two_point_count(catalog1=p, catalog2=random, n_bins=n_bins)
    RR = two_point_count(catalog1=random, catalog2=random, n_bins=n_bins)

    return (n_random_catalog / n_data)**2 * DD/RR - 2 * (n_random_catalog / n_data) * DD/DR + 1


def create_random_catalog(n_dimensions, n_particles):
    random_catalog = np.random.random((n_dimensions, n_particles))
    return random_catalog


def two_point_count(catalog1, catalog2, n_bins):
    # slow way:
    n_catalog1 = np.shape(catalog1[1])
    n_catalog2 = np.shape(catalog2[1])

    min_separation = 0
    max_separation = side_length / np.sqrt(2)

    bin_edges = np.linsapce(min_separation, max_separation, n_bins + 1)
    counts = np.zeros(n_bins + 1)

    # todo: stop double counting if catalog1 = catalog2
    for i in range(n_catalog1):
        distance_array = np.linalg.norm(catalog2 - catalog1[:, i])
        counts_for_i, _ = np.histogram(distance_array, bins=bin_edges)
        counts += counts_for_i

    return counts


# Define procedure to update positions at each timestep
def update(i):
    global position, velocity
    position += velocity * dt  # Increment positions according to their velocites
    position = apply_boundary(position)  # Apply boundary conditions
    # print("")
    # print(position, velocity)
    apply_force_pp(position, velocity)
    position, velocity = apply_force_pm(position, velocity)
    # print("")
    # print(position, velocity)
    points.set_data(position[1, :], position[2, :])  # Show 2D projection of first 2 position coordinates
    return points,


if __name__ == '__main__':
    # Set the axes on which the points will be shown
    plt.ion()  # Set interactive mode on
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-side_length * 0.5, side_length * 0.5)
    ax.set_ylim(-side_length * 0.5, side_length * 0.5)

    # Create command which will plot the positions of the particles
    points, = plt.plot([], [], 'o', markersize=5)

    ani = FuncAnimation(fig, update, frames=n_frames)# interval=frame_duration)
    f = r"simpp.mp4"
    writervideo = FFMpegWriter(fps=60)
    ani.save(f, writer=writervideo)


