import numpy as np


def power_spectrum(delta_ft, k_norm, n_bins):

    bin_edges = np.linspace(np.min(k_norm), np.max(k_norm), n_bins + 1)
    bin_mids = (bin_edges + (bin_edges[1] - bin_edges[0]))[:-1]
    P = np.zeros(n_bins)

    c1, c2, c3 = np.shape(delta_ft)
    for i in range(c1):
        for j in range(c2):
            for k in range(c3):
                k_val = k_norm[i, j, k]
                idx = np.argmin(np.abs(k_val - bin_mids))
                P[idx] += np.abs(delta_ft[i, j, k])**2

    return bin_mids, P
