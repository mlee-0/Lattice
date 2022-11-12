"""Dataset classes that load previously cached dataset files."""


import os
import random
import time
from typing import List, Tuple, Union

import numpy as np
import torch
# import torch_geometric

from preprocessing import DATASET_FOLDER, read_pickle
from visualization import *


class StrutDataset(torch.utils.data.Dataset):
    """Density and node location data as 2-channel 3D tensors and strut diameter data as scalars. The node location channel is a binary array with values {0, 1}.
    
    In this dataset, a single input image corresponds to many struts. Splitting this dataset for training and testing should be done by input image, instead of by strut, in order to ensure that struts corresponding to input images in the training set do not show up in the testing set.

    `count`: Number of input images to include in the dataset. All struts associated with these images are included.
    `p`: Proportion of data to include in the dataset. For example, if 0.1, 10% of the data are randomly sampled. Set a random seed before initializing this dataset to ensure reproducibility.
    `normalize_inputs`: Normalize input values to zero mean and unit variance.
    `struts`: List of strut coordinates ((x1, y1, z1), (x2, y2, z2)) to include in the dataset. If None, all struts are included.
    """

    def __init__(self, count: int=None, p: float=1.0, normalize_inputs: bool=True, struts: List[tuple]=None) -> None:
        super().__init__()
        self.p = p

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs.pickle'))

        if count is not None:
            self.inputs = self.inputs[:count, ...]
            self.outputs = [_ for _ in self.outputs if _[0] < count]
        
        if struts is not None:
            struts = [sorted(strut) for strut in struts]
            self.outputs = [_ for _ in self.outputs if sorted((_[1], _[2])) in struts]
        
        self.diameters = torch.tensor([output[3] for output in self.outputs])[:, None]
        
        # Normalize input data.
        if normalize_inputs:
            self.input_mean = self.inputs.mean()
            self.input_std = self.inputs.std()

            self.inputs -= self.input_mean
            self.inputs /= self.input_std

        # # Normzalize label data.
        # self.diameter_mean = self.diameters.mean().item()
        # self.diameter_std = self.diameters.std().item()
        # self.diameters -= self.diameter_mean
        # self.diameters /= self.diameter_std

        # Add a second channel for storing node locations. Values should not be normalized.
        self.inputs = torch.cat([self.inputs, torch.zeros_like(self.inputs)], dim=1)

    def __len__(self) -> int:
        return len(self.outputs)

    # Use with ResNetMasked
    # def __getitem__(self, index):
    #     # Return the coordinates of the two nodes in addition to the inputs and outputs.
    #     image_index, (x1, y1, z1), (x2, y2, z2), diameter = self.outputs[index]
    #     return torch.clone(self.inputs[image_index, :1, ...]), ((x1, y1, z1), (x2, y2, z2)), torch.clone(self.diameters[index, :])

    def __getitem__(self, index):
        image_index, (x1, y1, z1), (x2, y2, z2), diameter = self.outputs[index]
        self.inputs[image_index, 1, ...] = 0
        self.inputs[image_index, 1, x1, y1, z1] = 1
        self.inputs[image_index, 1, x2, y2, z2] = 1
        return torch.clone(self.inputs[image_index, ...]), torch.clone(self.diameters[index, :])
    
    def outputs_for_images(self, image_indices: List[int]):
        """Return output data indices corresponding to a list of image indices."""
        image_indices = set(image_indices)
        indices = [i for i, output in enumerate(self.outputs) if output[0] in image_indices]
        indices = random.sample(indices, round(self.p * len(indices)))
        return indices

    def normalize_inputs(self, x: torch.Tensor) -> None:
        """Normalize the given tensor of input values to zero mean and unit variance, in-place."""
        x -= self.input_mean
        x /= self.input_std

    def unnormalize_inputs(self, x: torch.Tensor) -> None:
        """Unnormalize the given tensor of input values to its original range, in-place."""
        x *= self.input_std
        x += self.input_mean

# class CenteredStrutDataset(torch.utils.data.Dataset):
#     """Only include 7 struts at the center of the volume per input data."""

#     def __init__(self, count: int=None, p: float=1.0, normalize_inputs: bool=True) -> None:
#         super().__init__()
#         self.p = p

#         self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs.pickle')).float() / 255
#         self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs.pickle'))

#         # Remove struts that are not at the center.
#         self.outputs = [
#             _ for _ in self.outputs if _[1] == (5, 5, 5) and _[2] in [
#                 # (6, 5, 5),
#                 # (5, 6, 5),
#                 # (5, 5, 6),
#                 # (6, 6, 5),
#                 # (5, 6, 6),
#                 (6, 5, 6),
#                 # (6, 6, 6),
#             ]
#         ]

#         if count is not None:
#             self.inputs = self.inputs[:count, ...]
#             self.outputs = [_ for _ in self.outputs if _[0] < count]
        
#         self.diameters = torch.tensor([output[3] for output in self.outputs])[:, None]
        
#         # Normalize input data.
#         if normalize_inputs:
#             self.input_mean = self.inputs.mean()
#             self.input_std = self.inputs.std()

#             self.inputs -= self.input_mean
#             self.inputs /= self.input_std

#         # # Normalize label data.
#         # self.diameter_mean = self.diameters.mean().item()
#         # self.diameter_std = self.diameters.std().item()
#         # self.diameters -= self.diameter_mean
#         # self.diameters /= self.diameter_std

#         # # Add a second channel for storing node locations. Values should not be normalized.
#         # self.inputs = torch.cat([self.inputs, torch.zeros_like(self.inputs)], dim=1)

#     def __len__(self) -> int:
#         return len(self.outputs)

#     def __getitem__(self, index):
#         image_index, (x1, y1, z1), (x2, y2, z2), diameter = self.outputs[index]
#         return torch.clone(self.inputs[image_index, ...]), torch.clone(self.diameters[index, :])
    
#     def outputs_for_images(self, image_indices: List[int]):
#         """Return output data indices corresponding to a list of image indices."""
#         image_indices = set(image_indices)
#         indices = [i for i, output in enumerate(self.outputs) if output[0] in image_indices]
#         indices = random.sample(indices, round(self.p * len(indices)))
#         return indices


# class GraphDataset(torch_geometric.data.Dataset):
#     """Density data and strut diameter data as graphs."""

#     def __init__(self, count: int=None) -> None:
#         self.dataset = read_pickle(os.path.join(DATASET_FOLDER, 'graphs.pickle'))
        
#         if count is not None:
#             self.dataset = self.dataset[:count]
        
#         # Normalize input data to have zero mean and unit variance.
#         inputs = torch.cat([graph.x.flatten() for graph in self.dataset])
#         mean, std = inputs.mean(), inputs.std()
#         for graph in self.dataset:
#             graph.x = graph.x[:, :1]  # Remove coordinate information
#             # graph.x -= mean
#             # graph.x /= std

#     def __len__(self) -> int:
#         return len(self.dataset)

#     def __getitem__(self, index):
#         # Return the entire graph. Returning individual attributes (x, edge_index, y) results in incorrect batching.
#         return self.dataset[index]

class InferenceDataset(torch.utils.data.Dataset):
    """A dataset for testing the iterative lattice generation process. The strut being predicted is fixed at the center of the density matrix."""

    def __init__(self, density_shape: Tuple[int, int, int], density_range: Tuple[int, int], density_function: str, lattice_shape: Tuple[int, int, int], lattice_type: str) -> None:
        super().__init__()
        assert lattice_type in ('rectangle', 'circle')

        from scipy.ndimage import gaussian_filter

        # Shape of density matrix.
        h, w, d = density_shape

        # Generate a density matrix with the specified type.
        if density_function == 'linear':
            self.density = np.ones((h, w, d))
            self.density *= np.concatenate([
                # np.zeros(d//8),
                np.linspace(density_range[0], density_range[1], d),
                # np.ones(d//8),
            ]) * 255
        
        elif density_function == 'sin':
            self.density = np.ones((h, w, d))
            self.density *= (np.sin(np.linspace(0, 2*np.pi, d)) * ((density_range[1]-density_range[0]) / 2) + 0.5) * 255
        
        elif density_function == 'cos':
            self.density = np.ones((h, w, d))
            self.density *= (np.cos(np.linspace(0, 2*np.pi, d)) * ((density_range[1]-density_range[0]) / 2) + 0.5) * 255
        
        elif density_function == 'exp':
            self.density = np.ones((h, w, d))
            self.density *= np.exp(np.linspace(density_range[0], density_range[1], d))
            self.density -= self.density.min()
            self.density /= self.density.max()
            self.density *= (density_range[1] - density_range[0])
            self.density += density_range[0]
            self.density *= 255
        
        elif density_function == 'random':
            # np.random.seed(42)
            self.density = np.random.rand(h, w, d)
            self.density = gaussian_filter(self.density, sigma=3)
            self.density -= self.density.min()
            self.density /= self.density.max()
            self.density *= (density_range[1] - density_range[0])
            self.density += density_range[0]
            self.density *= 255

        # visualize_input(self.density, opacity=1.0)

        # Normalize density values.
        self.density -= 127.4493
        self.density /= 41.9801
        self.density = torch.tensor(self.density)

        # Generate a lattice structure within the volume, as a list containing pairs of node coordinates.
        self.indices = []

        if lattice_type == 'rectangle':
            # List of struts to add at each node.
            struts = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))
            X, Y, Z = [np.arange(size) + (density_size - size) // 2 for density_size, size in zip(density_shape, lattice_shape)]

            for x in X:
                for y in Y:
                    for z in Z:
                        for dx, dy, dz in struts:
                            # Prevent adding invalid struts at the edges that are not connected on one end.
                            if x == X[-1] and dx > 0 or y == Y[-1] and dy > 0 or z == Z[-1] and dz > 0:
                                continue

                            self.indices.append((
                                (x, y, z),  # Node 1 for current strut
                                (x+dx, y+dy, z+dz),  # Node 2 for current strut
                            ))
        
        elif lattice_type == 'circle':
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

            for z in Z:
                for i in range(XY.shape[1] - 1):
                    x1, y1 = XY[:, i]
                    x2, y2 = XY[:, i+1]
                    self.indices.append((
                        (x1, y1, z),
                        (x2, y2, z),
                    ))

                    if z != Z[-1]:
                        self.indices.append((
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
                                    if (node_1, node_2) not in self.indices and (node_2, node_1) not in self.indices:
                                        self.indices.append((node_1, node_2))

        # Initialize the second channel, which contains the locations of the two nodes that form a strut.
        self.channel_2 = torch.zeros((11, 11, 11))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Coordinates of node 1.
        x, y, z = 5, 5, 5

        (x1, y1, z1), (x2, y2, z2) = self.indices[index]
        channel_1 = self.density[x1-x:x1+x+1, y1-y:y1+y+1, z1-z:z1+z+1]

        self.channel_2[:] = 0
        self.channel_2[x, y, z] = 1
        self.channel_2[x + (x2-x1), y + (y2-y1), z + (z2-z1)] = 1

        return torch.cat([channel_1[None, ...], self.channel_2[None, ...]], dim=0).float()


if __name__ == '__main__':
    dataset = InferenceDataset('circle')