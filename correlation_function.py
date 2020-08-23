import numpy as np

side_length = 1


def correlation_function_ls(p):
    n_bins = 20
    random_catalog_factor = 1

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
    n_catalog1 = np.shape(catalog1[1])
    n_catalog2 = np.shape(catalog2[1])

    min_separation = 0
    max_separation = side_length * np.sqrt(3)

    bin_edges = np.linsapce(min_separation, max_separation, n_bins + 1)
    distance_array = separation(catalog1, catalog2)
    counts, _ = np.histogram(np.flatten(distance_array), bins=bin_edges)
    return counts


def separation(catalog1, catalog2):
    s = catalog1[:, None, :] - catalog2[:, :, None]
    return np.sum(s**2, axis=0)**0.5
