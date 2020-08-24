import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.fft import fftn, ifftn, rfftn, irfftn
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files (x86)\\FFMpeg\\bin\\ffmpeg.exe'

import sys
sys.path.append("C:\\Users\\matth\\PycharmProjects\\PHYS4080\\GalaxySim")


np.random.seed(42)

# Sim setup
n_particles = 2
n_frames = 100
frame_duration = 100

n_grid_cells = 50

L_box = 1
H_0 = 1
G = 1
Omega_m0 = 1
mass_particle = 1
Omega_0 = 1

r_0 = L_box / n_grid_cells
t_0 = 1 / H_0
rho_0 = 3 * H_0**2 * Omega_m0 / (8 * np.pi * G)
v_0 = r_0 / t_0
phi_0 = v_0 ** 2


# Set initial positions and velocities
position = n_grid_cells * np.random.random((3, n_particles))
# position = n_grid_cells * (0.5 - np.array([np.linspace(0, 1, n_particles), np.linspace(0, 1, n_particles), np.zeros(n_particles)]))
velocity = 10 * (0.5 - np.random.random((3, n_particles)))

# position = np.array([[25., 35.], [0., 0.], [0., 0.]])
# velocity = np.array([[0., 0.], [10., 10.], [0., 0.]])

# frame counter
frame = 0

#
a = 0.1
da = 1e-2


def apply_boundary(p):
    """Defines the boundary conditions - we want a periodic boundary"""
    p = p % n_grid_cells
    return p


def assign_densities(p):
    rho = np.array(np.zeros((n_grid_cells, n_grid_cells, n_grid_cells), dtype=float))
    cell_centers = np.floor(p)
    cell_centers = cell_centers.astype(int)
    d = p - cell_centers
    t = 1 - d

    # assigning mass using CIC
    # might be able to vectorise somehow
    for i in range(n_particles):
        l, m, n = cell_centers[:, i]
        mp = mass_particle

        lp1 = (l + 1) % n_grid_cells
        mp1 = (m + 1) % n_grid_cells
        np1 = (n + 1) % n_grid_cells

        rho[l, m, n] += mp * t[0, i] * t[1, i] * t[2, i]
        rho[l, mp1, n] += mp * t[0, i] * d[1, i] * t[2, i]
        rho[l, m, np1] += mp * t[0, i] * t[1, i] * d[2, i]
        rho[l, mp1, np1] += mp * t[0, i] * d[1, i] * d[2, i]
        rho[lp1, m, n] += mp * d[0, i] * t[1, i] * t[2, i]
        rho[lp1, mp1, n] += mp * d[0, i] * d[1, i] * t[2, i]
        rho[lp1, m, np1] += mp * d[0, i] * t[1, i] * d[2, i]
        rho[lp1, mp1, np1] += mp * d[0, i] * d[1, i] * d[2, i]

    return rho


def apply_force_pm(p, v):
    rho = assign_densities(p)
    rhobar = (np.sum(rho) / n_grid_cells**3)
    delta = (rho - rhobar) / rhobar
    delta_ft = rfftn(delta, axes=(0, 1, 2)) #/ n_grid_cells**(2/3)

    phi_ft = delta_ft * greens
    phi = irfftn(phi_ft, axes=(0, 1, 2))

    g_field_slow = get_acceleration_field(phi)

    g_particle = assign_accelerations(p, g_field_slow)

    if plot_powerspectrum:

        k_bin, P = power_spectrum(delta_ft, k_norm, n_bins=50, delta=delta)
        r, xi = correlation_function(k_bin, P)
        r = r * n_grid_cells

        powspec.set_data(k_bin, P)
        ax2.set_ylim([0, np.max(P) * 1.2])

        corrfunc.set_data(range(len(xi)), xi)
        ax3.set_ylim([np.min(xi) * 1.2, np.max(xi) * 1.2])

    v += g_particle * da

    return v


def assign_accelerations(p, g_field):
    g = np.array(np.zeros((3, n_particles), dtype=float))
    cell_centers = np.floor(p)
    cell_centers = cell_centers.astype(int)
    d = p - cell_centers
    t = 1 - d

    # assigning acceleration using CIC
    # might be able to vectorise this somehow
    for i in range(n_particles):
        l, m, n = cell_centers[:, i]

        lp1 = (l + 1) % n_grid_cells
        mp1 = (m + 1) % n_grid_cells
        np1 = (n + 1) % n_grid_cells

        g[:, i] = g_field[:, l, m, n] * t[0, i] * t[1, i] * t[2, i] \
                  + g_field[:, l, mp1, n] * t[0, i] * d[1, i] * t[2, i] \
                  + g_field[:, l, m, np1] * t[0, i] * t[1, i] * d[2, i] \
                  + g_field[:, l, mp1, np1] * t[0, i] * d[1, i] * d[2, i] \
                  + g_field[:, lp1, m, n] * d[0, i] * t[1, i] * t[2, i] \
                  + g_field[:, lp1, mp1, n] * d[0, i] * d[1, i] * t[2, i] \
                  + g_field[:, lp1, m, np1] * d[0, i] * t[1, i] * d[2, i] \
                  + g_field[:, lp1, mp1, np1] * d[0, i] * d[1, i] * d[2, i]

    return g


def get_acceleration_field(phi):
    g = np.zeros((3, n_grid_cells, n_grid_cells, n_grid_cells), dtype=float)

    for i in range(n_grid_cells):
        ip1 = (i + 1) % n_grid_cells
        im1 = (i - 1) % n_grid_cells
        g[0, i, :, :] = -(phi[ip1, :, :] - phi[im1, :, :]) / 2

    for j in range(n_grid_cells):
        jp1 = (j + 1) % n_grid_cells
        jm1 = (j - 1) % n_grid_cells
        g[1, :, j, :] = -(phi[:, jp1, :] - phi[:, jm1, :]) / 2

    for k in range(n_grid_cells):
        kp1 = (k + 1) % n_grid_cells
        km1 = (k - 1) % n_grid_cells
        g[2, :, :, k] = -(phi[:, :, kp1] - phi[:, :, km1]) / 2

    return g


def greens_function(k_vec):
    greens = -3 * Omega_0 / (8 * a * np.linalg.norm(np.sin(k_vec/2), axis=0))
    greens[0, 0, 0] = 0
    return greens


### ~~~ Power Spectrum ~~~ ###


def power_spectrum(delta_ft, k_norm, n_bins, delta):
    bin_edges = np.linspace(np.min(k_norm), np.max(k_norm), n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_mids = (bin_edges + bin_width)[:-1]

    alpha = 1

    F_rvec = delta - alpha * delta_random
    F_kvec = rfftn(F_rvec, axes=(0, 1, 2))

    # F_kvec = delta_ft #- alpha * delta_ft_random
    P_kvec = F_kvec * np.conj(F_kvec) #/ (2 * np.pi) ** 6
    P = np.zeros(n_bins)

    for i in range(n_bins):
        condition = np.where(np.abs(k_norm - bin_mids[i]) < bin_width/2)
        print(np.sum(condition))
        P[i] = np.sum(P_kvec[condition]) / np.sum(condition)
    # plt.close()
    # plt.plot(bin_mids, P)
    # plt.savefig('volume.png')
    # exit()


    return bin_mids, P


def get_random_catalogue(alpha, n_sim):
    rand_pos = np.random.random((3, alpha * n_sim))
    rand_density = assign_densities(rand_pos)
    return rand_density


def correlation_function(k, P):
    xi = np.fft.irfft(P)
    # print(xi)
    r = k / (2 * np.pi)
    # print(r)
    half = int(len(P))
    xi = xi[half:]
    # xi = savgol_filter(xi, 7, 3)

    return r, xi


def correlation_function_nsquared(p):


def update(i):
    global position, velocity, frame, a

    if frame % 10 == 0:
        print(frame)

    position += velocity * da
    position = apply_boundary(position)
    velocity = apply_force_pm(position, velocity)

    points_sim.set_data(position[0, :], position[1, :])

    a += da
    frame += 1

    return points_sim,


if __name__ == '__main__':

    plot_powerspectrum = True

    x = np.arange(0, n_grid_cells)
    pos_grid = np.array(np.meshgrid(x, x, x))
    k_grid = 2 * np.pi * pos_grid / n_grid_cells
    k_norm = np.linalg.norm(k_grid, axis=0)

    half = int(n_grid_cells / 2 + 1)
    k_grid = k_grid[:, :, :, :half]
    k_norm = k_norm[:, :, :half]

    greens = greens_function(k_grid)

    alpha = 10
    rho_random = get_random_catalogue(alpha, n_particles)
    rhobar_random = (np.sum(rho_random) / n_grid_cells ** 3)
    delta_random = (rho_random - rhobar_random) / rhobar_random
    delta_ft_random = rfftn(delta_random, axes=(0, 1, 2))  # / n_grid_cells**(2/3)

    if plot_powerspectrum:
        plt.ion()

        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)
        ax1 = fig.add_subplot(gs[0:, 0:2])
        ax1.set_title('Main Simulation')

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('Power Spectrum')

        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_title('Correlation Function')

        ax1.set_xlim(0, n_grid_cells)
        ax1.set_ylim(0, n_grid_cells)

        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)

        ax3.set_xlim(0, 50)
        ax3.set_ylim(0, 10)

        # Create command which will plot the positions of the particles
        points_sim, = ax1.plot([], [], 'o', markersize=1)
        powspec, = ax2.plot([], [], markersize=1)
        corrfunc, = ax3.plot([], [], markersize=1)

        ani = FuncAnimation(fig, update, frames=n_frames)  # interval=frame_duration)
        f = r"sim_pm.mp4"
        writervideo = FFMpegWriter(fps=60)
        ani.save(f, writer=writervideo)

    if not plot_powerspectrum:


        # Set the axes on which the points will be shown
        plt.ion()  # Set interactive mode on
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, n_grid_cells)
        ax.set_ylim(0, n_grid_cells)

        # Create command which will plot the positions of the particles
        points_sim, = plt.plot([], [], 'o', markersize=1)

        ani = FuncAnimation(fig, update, frames=n_frames)# interval=frame_duration)
        f = r"sim_pm.mp4"
        writervideo = FFMpegWriter(fps=60)
        ani.save(f, writer=writervideo)


