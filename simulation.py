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
n_particles = 1000
n_frames = 2000
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
G = 1000                           # m^3 kg^-1 s^-2
v_max = 0.0                         # m/s
side_length = 1                # in units of m
radius_buffer = 0.2 * side_length           # in units of m
dt = 0.005                         # in units of seconds
omega_0 = 1
cosmo_a = 1
cosmo_f = 1

# Set initial positions and velocities
position = side_length * (0.5 - np.random.random((n_dimensions, n_particles)))
# position = side_length * (0.5 - np.array([np.linspace(0, 1, n_particles), np.linspace(0, 1, n_particles), np.linspace(0, 1, n_particles)]))
velocity = v_max * (0.5 - np.random.random((n_dimensions, n_particles)))

# frame counter
frame = 0


def apply_boundary(p):
    """Defines the boundary conditions - we want a periodic boundary"""
    p[p > side_length/2] -= side_length
    p[p < -side_length/2] += side_length
    return p


def apply_force_pp(p, v):

    for m in range(n_particles):
        # particle-particle interaction
        distance_factor = 0
        for n in range(n_particles):
            if n == m:
                continue

            distance_factor += pp_displacement(p[:, m], p[:, n], softening_length=radius_buffer)
        # this might also need some 4s and pis etc.
        v[:, m] += - G * particle_mass * distance_factor * dt
    return p, v


def pp_displacement(p1, p2, softening_length):
    return (p1 - p2) / (np.linalg.norm(p1 - p2) ** n_dimensions + softening_length)


def apply_force_pm(p, v):
    cell_num = 50
    particle_coords = get_particle_coords(p, cell_num)
    rho = ngp(particle_coords, cell_num)
    rhobar = np.sum(rho) / side_length ** 3
    delta = (rho - rhobar) / rhobar
    delta_ft = rfftn(delta, axes=(0, 1, 2))


    x = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1] * side_length
    pos_grid = np.array(np.meshgrid(x, x, x))
    k_grid = 2 * np.pi * pos_grid / side_length
    k_norm = np.linalg.norm(k_grid, axis=0)

    # halving bc rfft is funny
    half = int(cell_num / 2 + 1)
    k_grid = k_grid[:, :, :, :half]
    k_norm = k_norm[:, :, :half]

    phi_ft = greens_function(k_grid) * delta_ft

    # phi = irfftn(phi_ft, axes=(0, 1, 2))

    a_ft_grid = 4 * 1j * np.pi * G * phi_ft * k_grid
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


def apply_force_p3m(p, v):
    cell_num = 50
    particle_coords = get_particle_coords(p, cell_num)

    # assigning into grid
    all_cells = np.empty([cell_num, cell_num, cell_num], dtype=list)

    for i in range(n_particles):
        particle_coord = particle_coords[:, i]
        if all_cells[tuple(particle_coord)] is None:
            all_cells[tuple(particle_coord)] = [i]
        else:
            all_cells[tuple(particle_coord)] += [i]

    rho = ngp(particle_coords, cell_num)
    rhobar = np.sum(rho) / side_length**3
    delta = (rho - rhobar) / rhobar
    delta_ft = rfftn(delta, axes=(0, 1, 2))


    x = (np.linspace(-0.5, 0.5, cell_num + 1) + 0.5 / cell_num)[:-1] * side_length
    pos_grid = np.array(np.meshgrid(x, x, x))
    k_grid = 2 * np.pi * pos_grid / side_length
    k_norm = np.linalg.norm(k_grid, axis=0)



    # halving bc rfft is funny
    half = int(cell_num / 2 + 1)
    k_grid = k_grid[:, :, :, :half]
    k_norm = k_norm[:, :, :half]

    a_ft_grid = 4 * 1j * np.pi * G * delta_ft * k_grid / k_norm**2
    a_grid = irfftn(a_ft_grid, axes=(1, 2, 3))

    # power_spectrum_from_density_ft(delta_ft, k_norm)

    # to_plot = np.sum(np.linalg.norm(a_grid, axis=0), axis=2)
    #
    # plt.close()
    # plt.imshow(to_plot)
    # plt.savefig('rho.png')
    #
    # exit()



    for j in range(n_particles):
        j_coords = particle_coords[:, j]
        pm_accel = a_grid[:, j_coords[0], j_coords[1], j_coords[2]]

        cell_depth = 3

        local_cells = np.array([[j_coords[0] + l, j_coords[1] + m, j_coords[2] + n]
                       for l in range(-cell_depth, cell_depth, 1)
                       for m in range(-cell_depth, cell_depth, 1)
                       for n in range(-cell_depth, cell_depth, 1)])

        local_cells[local_cells >= 50] -= 50
        local_cells[local_cells <= -1] += 50

        distance_factor = 0
        for f in range(np.shape(local_cells)[1]):
            cell_idx = tuple(local_cells[f])
            if all_cells[cell_idx] is None:
                continue
            else:
                nearby = all_cells[cell_idx]
                cell_pos = ((local_cells[f]) / cell_num - 0.5) * side_length
                for nb in nearby:
                    distance_factor += pp_displacement(p[:, j], p[:, nb], softening_length=radius_buffer)
                    distance_factor -= pp_displacement(p[:, j], cell_pos, softening_length=radius_buffer)

        pp_accel = - G * particle_mass * distance_factor
        v[:, j] += (pm_accel + pp_accel) * dt

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


def correlation_function_ls(p, bin_edges):
    random_catalog_factor = 1

    n_data = np.shape(p)[1]
    n_random_catalog = random_catalog_factor * n_data

    random = create_random_catalog(n_dimensions, n_random_catalog)
    DD = two_point_count(catalog1=p, catalog2=p, bin_edges=bin_edges)
    DR = two_point_count(catalog1=p, catalog2=random, bin_edges=bin_edges)
    RR = two_point_count(catalog1=random, catalog2=random, bin_edges=bin_edges)

    corr_ls = (n_random_catalog / n_data)**2 * DD/RR - 2 * (n_random_catalog / n_data) * DD/DR + 1
    corr_ls[np.isnan(corr_ls)] = 0
    corr_ls[np.isinf(corr_ls)] = 0
    # corr_ls[corr_ls < -0.999] = 0

    return corr_ls


def correlation_function_ph(p, bin_edges):
    random_catalog_factor = 10

    n_data = np.shape(p)[1]
    n_random_catalog = random_catalog_factor * n_data

    random = create_random_catalog(n_dimensions, n_random_catalog)
    DD = two_point_count(catalog1=p, catalog2=p, bin_edges=bin_edges)
    RR = two_point_count(catalog1=random, catalog2=random, bin_edges=bin_edges)

    corr_ph = (n_random_catalog / n_data)**2 * DD/RR - 1
    corr_ph[np.isnan(corr_ph)] = 0
    corr_ph[np.isinf(corr_ph)] = 0
    corr_ph[corr_ph < -0.999] = 0
    return corr_ph, DD


def create_random_catalog(n_dimensions, n_particles):
    random_catalog = np.random.random((n_dimensions, n_particles))
    return random_catalog


def two_point_count(catalog1, catalog2, bin_edges):
    distance_array = separation(catalog1, catalog2)
    counts, _ = np.histogram(distance_array, bins=bin_edges)
    return counts


def separation(catalog1, catalog2):
    s = catalog1[:, None, :] - catalog2[:, :, None]

    sx = np.triu(s[0], k=1)
    sx = sx[np.nonzero(sx)]

    sy = np.triu(s[0], k=1)
    sy = sy[np.nonzero(sy)]

    sz = np.triu(s[0], k=1)
    sz = sz[np.nonzero(sz)]

    s = np.array([sx, sy, sz])
    s = np.linalg.norm(s, axis=0)

    return s


def power_spectrum_from_density_ft(density_ft, k_norm):

    print(np.shape(density_ft))
    print(np.shape(k_norm))

    n_bins = 100
    bin_edges = np.linspace(np.min(k_norm), np.max(k_norm), n_bins + 1)
    bin_mids = (bin_edges + (bin_edges[1] - bin_edges[0]))[:-1]
    P = np.zeros(n_bins)

    c1, c2, c3 = np.shape(density_ft)
    for i in range(c1):
        for j in range(c2):
            for k in range(c3):
                k_val = k_norm[i, j, k]
                idx = np.argmin(np.abs(k_val - bin_mids))
                P[idx] += np.abs(density_ft[i, j, k])**2

    plt.close()
    print(P)
    plt.plot(bin_mids, P)
    plt.savefig('power_spectrum')
    exit()
    return


def greens_function(k_vec):
    greens = -3 * omega_0 / (8 * cosmo_a) * np.linalg.norm(np.sin(k_vec), axis=0) ** -1
    greens[0, 0, 0] = 0
    return greens



def update(i):
    global position, velocity, frame

    frame += 1
    # if frame == 2:
    #
    #     min_separation = 0
    #     max_separation = side_length * np.sqrt(3)
    #     n_bins = 100
    #     bin_edges = np.linspace(min_separation, max_separation, n_bins + 1)
    #     bin_middles = (bin_edges + bin_edges[1] - bin_edges[0])[:-1]
    #     plt.close()
    #     # plt.plot(bin_middles, correlation_function_ls(position, bin_edges))
    #     corr_ph, dd = correlation_function_ph(position, bin_edges)
    #     plt.plot(bin_middles, dd)
    #     plt.savefig('dd.png')
    #     plt.close()
    #     plt.plot(bin_middles, corr_ph)
    #     plt.savefig('corrfunc.png')
    #
    #     exit()
    #     print(frame)

    if frame % 10 == 0:
        print(frame)

    position += velocity * dt
    position = apply_boundary(position)
    position, velocity = apply_force_pm(position, velocity)

    points.set_data(position[0, :], position[1, :])


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
    f = r"sim_pm.mp4"
    writervideo = FFMpegWriter(fps=60)
    ani.save(f, writer=writervideo)


