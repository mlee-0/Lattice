"""
Functions for generating custom densities and lattice structures.
"""


import os
from typing import *

import numpy as np

from preprocessing import DATASET_FOLDER, DIRECTIONS, read_pickle


def generate_linear_density(minimum: float, maximum: float, h: int, w: int, d: int):
    density = np.ones((h, w, d))
    density *= np.concatenate([
        # np.zeros(d//8),
        np.linspace(minimum, maximum, d),
        # np.ones(d//8),
    ])
    return density

def generate_sinusoidal_density(minimum: float, maximum: float, h: int, w: int, d: int):
    amplitude = (maximum - minimum) / 2
    density = np.ones((h, w, d))
    density *= np.sin(np.linspace(0, 2*np.pi, d)) * amplitude + 0.5
    return density

def generate_exponential_density(minimum: float, maximum: float, h: int, w: int, d: int):
    density = np.ones((h, w, d))
    density *= np.exp(np.linspace(minimum, maximum, d))
    density -= density.min()
    density /= density.max()
    density *= (maximum - minimum)
    density += minimum
    return density

def generate_random_density(minimum: float, maximum: float, h: int, w: int, d: int):
    from scipy.ndimage import gaussian_filter

    # np.random.seed(42)
    density = np.random.rand(h, w, d)
    density = gaussian_filter(density, sigma=3)
    density -= density.min()
    density /= density.max()
    density *= (maximum - minimum)
    density += minimum
    return density

def load_stress_density(*args):
    density = read_pickle(os.path.join(DATASET_FOLDER, 'stress.pickle'))
    density -= density.min()
    density /= density.max()

    # density = np.pad(density, (6, 6), mode='maximum')

    return density

def load_topology_optimization_density(*args):
    density = read_pickle(os.path.join(DATASET_FOLDER, 'topology_optimization.pickle'))
    density -= density.min()
    density /= density.max()

    density = np.pad(density, (20, 20), mode='edge')
    return density

def generate_rectangular_lattice(density_shape: Tuple[int, int, int], lattice_shape: Tuple[int, int, int]):
    
    x_coordinates, y_coordinates, z_coordinates = [np.arange(size) + (density_size - size) // 2 for density_size, size in zip(density_shape, lattice_shape)]

    indices = []

    for x in x_coordinates:
        for y in y_coordinates:
            for z in z_coordinates:
                for dx, dy, dz in DIRECTIONS:
                    # Prevent adding invalid struts at the edges that are not connected on one end.
                    if x == x_coordinates[-1] and dx > 0 or y == y_coordinates[-1] and dy > 0 or z == z_coordinates[-1] and dz > 0:
                        continue

                    indices.append((
                        (x, y, z),  # Node 1 for current strut
                        (x+dx, y+dy, z+dz),  # Node 2 for current strut
                    ))

    return indices

def generate_circular_lattice(density_shape: Tuple[int, int, int], lattice_shape: Tuple[int, int, int]):
    h, w, d = density_shape

    radius = max(lattice_shape[:2]) // 2
    theta = np.linspace(0, 360, 500) * (np.pi / 180)
    X = np.round(radius * np.cos(theta)).astype(int)
    Y = np.round(radius * np.sin(theta)).astype(int)
    # Delete horizontal/vertical struts that should instead be diagonal struts, to make the overall appearance more curved by introducing diagonal struts.
    outside_radius = np.sqrt(X ** 2 + Y ** 2) > radius
    X = X[outside_radius]
    Y = Y[outside_radius]
    # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    XY, index = np.unique(np.array([X, Y]), axis=1, return_index=True)
    XY = XY[:, np.argsort(index)]
    # Ensure the last node connects to the first.
    XY = np.append(XY, XY[:, :1], axis=1)
    # Center the X and Y coordinates within the range instead of at the origin.
    XY[0, :] += (h//2)
    XY[1, :] += (w//2)

    Z = np.arange(lattice_shape[2]) + (density_shape[2] - lattice_shape[2]) // 2

    indices = []
    for z in Z:
        for i in range(XY.shape[1] - 1):
            x1, y1 = XY[:, i]
            x2, y2 = XY[:, i+1]
            indices.append((
                (x1, y1, z),
                (x2, y2, z),
            ))

            if z != Z[-1]:
                indices.append((
                    (x1, y1, z),
                    (x1, y1, z+1),
                ))
        
        # Add horizontal/vertical struts inside the circle.
        for x in range(h):
            for y in range(w):
                if np.sqrt((x-h/2) ** 2 + (y-w/2) ** 2) <= radius:
                    node_1 = (x, y, z)
                    for node_2 in [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z)]:
                        if np.sqrt((node_2[0]-h/2) ** 2 + (node_2[1]-w/2) ** 2) <= radius:
                            if (node_1, node_2) not in indices and (node_2, node_1) not in indices:
                                indices.append((node_1, node_2))

    return indices

def generate_random_lattice(density_shape: Tuple[int, int, int], lattice_shape: Tuple[int, int, int]):
    import random

    h, w, d = density_shape

    min_coordinates = [(size - lattice_size) // 2 for size, lattice_size in zip([h,w,d], lattice_shape)]
    max_coordinates = [minimum + lattice_size - 1 for minimum, lattice_size in zip(min_coordinates, lattice_shape)]

    indices = []
    node_current = (h // 2, w // 2, d // 2)
    while random.random() > 1e-4:
        node_new = [coordinate + offset for coordinate, offset in zip(node_current, random.choices([-1, 0, 1], k=3))]
        # Check that the new node is not out of bounds.
        if all(min_ <= coordinate <= max_ for coordinate, min_, max_ in zip(node_new, min_coordinates, max_coordinates)):
            # Check if the new strut is a duplicate.
            if (node_current, node_new) not in indices and (node_new, node_current) not in indices:
                indices.append((node_current, node_new))
                node_current = node_new

    return indices