"""Dataset classes that load previously cached dataset files."""


import os
import random
from typing import *

import numpy as np
import torch
# import torch_geometric

from preprocessing import DATASET_FOLDER, read_pickle
from visualization import *


class LatticeDataset(torch.utils.data.Dataset):
    # Maximum value to scale diameters to.
    DIAMETER_SCALE = 100

    def __init__(self, normalize_inputs: bool=True) -> None:
        super().__init__()

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs_augmented.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs_array_augmented.pickle')).float()

        # Add a second binary channel with values of either -1 (inactive nodes) or 1 (active nodes).
        self.inputs = torch.cat(
            (self.inputs, torch.any(self.outputs > 0, dim=1, keepdim=True) * 2 - 1),
            dim=1,
        )

        # Normalize input data.
        if normalize_inputs:
            self.input_mean = self.inputs.mean()
            self.input_std = self.inputs.std()

            self.inputs -= self.input_mean
            self.inputs /= self.input_std
        
        # Scale label data.
        self.outputs *= self.DIAMETER_SCALE

        # Print a summary of the data.
        print(f"\nDataset '{type(self)}':")
        
        print(f"Input data:")
        print(f"\tShape: {self.inputs.size()}")
        print(f"\tData type: {self.inputs.dtype}")
        print(f"\tMemory: {self.inputs.storage().nbytes()/1e6:,.2f} MB")
        print(f"\tMin, max: {self.inputs.min()}, {self.inputs.max()}")
        print(f"\tMean, standard deviation: {self.inputs.mean():.2f}, {self.inputs.std():.2f}")

        print(f"Label data:")
        print(f"\tShape: {self.outputs.size()}")
        print(f"\tData type: {self.outputs.dtype}")
        print(f"\tMemory: {self.outputs.storage().nbytes()/1e6:,.2f} MB")
        print(f"\tMin, max: {self.outputs.min()}, {self.outputs.max()}")
        print(f"\tMean, standard deviation: {self.outputs.mean():.2f}, {self.outputs.std():.2f}")

    def __len__(self) -> int:
        return self.outputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index, ...], self.outputs[index, ...]

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

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs_augmented.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs_augmented.pickle'))

        if count is not None:
            self.inputs = self.inputs[:count, ...]
            self.outputs = [_ for _ in self.outputs if _[0] < count]
        
        if struts is not None:
            struts = [sorted(list(_) for _ in strut) for strut in struts]
            self.outputs = [_ for _ in self.outputs if sorted((_[1], _[2])) in struts]
        
        # # Remove struts within a certain distance from the X=0 and Y=0 edges.
        # self.outputs = [_ for _ in self.outputs if not any(coordinate in (0, 1) for coordinate in (_[1][:2] + _[2][:2]))]

        # Keep only struts at the center.
        center_struts = [
            [[5, 5, 5], [5, 5, 6]],
            [[5, 5, 4], [5, 5, 6]],
            [[5, 5, 5], [5, 6, 5]],
            [[5, 5, 5], [5, 6, 6]],
            [[5, 5, 4], [5, 6, 6]],
            [[5, 4, 5], [5, 6, 5]],
            [[5, 4, 5], [5, 6, 6]],
            [[5, 4, 4], [5, 6, 6]],
            [[5, 5, 5], [6, 5, 5]],
            [[5, 5, 5], [6, 5, 6]],
            [[5, 5, 4], [6, 5, 6]],
            [[5, 5, 5], [6, 6, 5]],
            [[5, 5, 5], [6, 6, 6]],
            [[5, 5, 4], [6, 6, 6]],
            [[5, 4, 5], [6, 6, 5]],
            [[5, 4, 5], [6, 6, 6]],
            [[5, 4, 4], [6, 6, 6]],
            [[4, 5, 5], [6, 5, 5]],
            [[4, 5, 5], [6, 5, 6]],
            [[4, 5, 4], [6, 5, 6]],
            [[4, 5, 5], [6, 6, 5]],
            [[4, 5, 5], [6, 6, 6]],
            [[4, 5, 4], [6, 6, 6]],
            [[4, 4, 5], [6, 6, 5]],
            [[4, 4, 5], [6, 6, 6]],
            [[4, 4, 4], [6, 6, 6]],
        ]
        self.outputs = [_ for _ in self.outputs if [_[1], _[2]] in center_struts]

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

    def __getitem__(self, index):
        """If using ResNetMasked, return the coordinates of the two nodes in addition to the inputs and outputs."""
        image_index, (x1, y1, z1), (x2, y2, z2), diameter = self.outputs[index]
        coordinates = ((x1, y1, z1), (x2, y2, z2))
        self.inputs[image_index, 1, ...] = 0
        self.inputs[image_index, 1, x1, y1, z1] = 1
        self.inputs[image_index, 1, x2, y2, z2] = 1
        return torch.clone(self.inputs[image_index, ...]), torch.clone(self.diameters[index, :]), coordinates

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

    def __init__(self, density_shape: Tuple[int, int, int], density_function: str, lattice_shape: Tuple[int, int, int], lattice_type: str) -> None:
        super().__init__()

        # Shape of density matrix.
        h, w, d = density_shape
        # Start and end values of density matrix.
        density_range = [0.0, 1.0]

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
            from scipy.ndimage import gaussian_filter

            # np.random.seed(42)
            self.density = np.random.rand(h, w, d)
            self.density = gaussian_filter(self.density, sigma=3)
            self.density -= self.density.min()
            self.density /= self.density.max()
            self.density *= (density_range[1] - density_range[0])
            self.density += density_range[0]
            self.density *= 255
        
        elif density_function == 'stress':
            self.density = read_pickle(os.path.join(DATASET_FOLDER, 'stress.pickle'))
            self.density -= self.density.min()
            self.density /= self.density.max()
            self.density *= 255

            self.density = np.pad(self.density, (6, 6), mode='maximum')
            density_shape = self.density.shape

            # from main import load_model
            # from models import Nie

            # density_shape = (15, 30, 15)

            # h, w, d = 20, 40, 30
            # input_data = np.zeros((1, 4, 20, 40))
            # input_data[:, 0, :h, :w] = 255
            # input_data[:, 1, :h, :d] = 255
            # input_data[:, 2, :h//2, w//2] = 255
            # input_data[:, 3, h//2, w//2:] = 255
            # # plt.figure()
            # # plt.subplot(2, 2, 1); plt.imshow(input_data[0, 0, ...])
            # # plt.subplot(2, 2, 2); plt.imshow(input_data[0, 1, ...])
            # # plt.subplot(2, 2, 3); plt.imshow(input_data[0, 2, ...])
            # # plt.subplot(2, 2, 4); plt.imshow(input_data[0, 3, ...])
            # # plt.show()
            # input_data = torch.tensor(input_data).float()
            # # Normalize based on the mean and standard deviation of the original dataset.
            # input_data -= 42.1984
            # input_data /= 94.2048

            # checkpoint = load_model(os.path.join(DATASET_FOLDER, 'stress_model.pth'), 'cpu')
            # stress_model = Nie(4, (20, 40), 15)
            # stress_model.load_state_dict(checkpoint['model_state_dict'])
            # stress_model.train(False)
            # with torch.no_grad():
            #     self.density = stress_model(input_data)
            #     self.density **= 1 / (1/1.8)
            #     # Scale to (_, 1) to exaggerate the contrast.
            #     self.density /= self.density.max()
            #     self.density *= 255
            # self.density = self.density[0, ...].numpy().transpose((1, 2, 0))
        
        elif density_function == 'topology opt.':
            self.density = read_pickle(os.path.join(DATASET_FOLDER, 'topology_optimization.pickle'))
            self.density -= self.density.min()
            self.density /= self.density.max()
            self.density *= 255

            self.density = np.pad(self.density, (20, 20), mode='edge')

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
        
        elif lattice_type == 'random':
            min_coordinates = [(size - lattice_size) // 2 for size, lattice_size in zip([h,w,d], lattice_shape)]
            max_coordinates = [minimum + lattice_size - 1 for minimum, lattice_size in zip(min_coordinates, lattice_shape)]

            node_current = (h // 2, w // 2, d // 2)
            while random.random() > 1e-4:
                node_new = [coordinate + offset for coordinate, offset in zip(node_current, random.choices([-1, 0, 1], k=3))]
                # Check that the new node is not out of bounds.
                if all(min_ <= coordinate <= max_ for coordinate, min_, max_ in zip(node_new, min_coordinates, max_coordinates)):
                    # Check if the new strut is a duplicate.
                    if (node_current, node_new) not in self.indices and (node_new, node_current) not in self.indices:
                        self.indices.append((node_current, node_new))
                        node_current = node_new
        
        else:
            raise NotImplementedError()

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
    # dataset = InferenceDataset('circle')
    dataset = LatticeDataset(True)