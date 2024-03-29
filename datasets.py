"""Dataset classes that load previously cached dataset files."""


import os
import random
from typing import *

import numpy as np
import torch

from inference import *
from preprocessing import DATASET_FOLDER, read_pickle
from visualization import *


class LatticeDataset(torch.utils.data.Dataset):
    # Maximum value to scale diameters to.
    DIAMETER_SCALE = 100

    def __init__(self, size: int=None, normalize_inputs: bool=True) -> None:
        super().__init__()

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs_augmented.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs_array_augmented.pickle')).float()

        if size is not None:
            self.inputs = self.inputs[:size]
            self.outputs = self.outputs[:size]

        # Add a second binary channel with values of either 0 (inactive nodes) or 1 (active nodes).
        self.inputs = torch.cat(
            (self.inputs, torch.any(self.outputs > 0, dim=1, keepdim=True)),
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

class AutoencoderDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs_augmented.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs_array_augmented.pickle')).float()

        self.inputs = self.inputs[:1000, ...]
        self.outputs = self.outputs[:1000, ...]

        # Add a second binary channel with values of either -1 (inactive nodes) or 1 (active nodes).
        self.inputs = torch.cat(
            (self.inputs, torch.any(self.outputs > 0, dim=1, keepdim=True) * 2 - 1),
            dim=1,
        )

        # Normalize input data.
        self.input_mean = self.inputs.mean()
        self.input_std = self.inputs.std()

        self.inputs -= self.input_mean
        self.inputs /= self.input_std

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index, ...], self.inputs[index, ...]

class FemurDataset(torch.utils.data.Dataset):
    """Relative density data in the shape of a femur, model from: https://link.springer.com/article/10.1007/s10237-015-0740-7"""

    def __init__(self, normalize_inputs: bool=True) -> None:
        super().__init__()

        shape = (30, 30, 100)

        channel_1 = read_pickle('Training_Data_50/inputs.pickle').float()[:1, :1, :shape[0], :shape[1], :shape[2]]
        channel_1 /= channel_1.max()
        # Enlarge in each dimension by repeating values to match `shape`.
        if channel_1.shape[2] < shape[0]:
            channel_1 = np.repeat(channel_1, np.ceil(shape[0]/channel_1.shape[2]), axis=2)
        if channel_1.shape[3] < shape[1]:
            channel_1 = np.repeat(channel_1, np.ceil(shape[1]/channel_1.shape[3]), axis=3)
        if channel_1.shape[4] < shape[2]:
            channel_1 = np.repeat(channel_1, np.ceil(shape[2]/channel_1.shape[4]), axis=4)
        # Index to match `shape`.
        channel_1 = channel_1[..., :shape[0], :shape[1], :shape[2]]

        channel_2 = np.zeros((1, 1, *shape))

        with open('Geraldes2015orthotropicfemurmodel.txt', 'r') as f:
            lines = f.readlines()[12:]

        x, y, z = zip(*[[float(_.strip()) for _ in line.split(',')[1:]] for line in lines])
        x, y, z = np.array(x), np.array(y), np.array(z)
        # Scale all coordinates within the bounds defined by `shape`.
        x -= x.min()
        y -= y.min()
        z -= z.min()
        maximum_coordinate = np.max(np.concatenate([x, y, z]))
        x = np.round((x / maximum_coordinate) * (max(shape) - 1)).astype(int)
        y = np.round((y / maximum_coordinate) * (max(shape) - 1)).astype(int)
        z = np.round((z / maximum_coordinate) * (max(shape) - 1)).astype(int)

        for x_, y_, z_ in zip(x, y, z):
            channel_2[..., x_, y_, z_] = 1
        channel_2 = channel_2 * 2 - 1
        
        self.inputs = np.concatenate([channel_1, channel_2], axis=1)
        self.inputs = torch.tensor(self.inputs).float()
        # Normalize input data.
        if normalize_inputs:
            self.inputs -= -0.14252009987831116
            self.inputs /= 0.7857609987258911

    def __getitem__(self, index):
        return self.inputs[index, ...]

class LatticeInferenceDataset(torch.utils.data.Dataset):
    """A dataset for testing the network on custom-generated density."""

    density_functions = {
        'linear': generate_linear_density,
        'sin': generate_sinusoidal_density,
        'exp': generate_exponential_density,
        'random': generate_random_density,
        'stress': load_stress_density,
        'top. opt.': load_topology_optimization_density,
    }

    def __init__(self, density_shape: Tuple[int, int, int], density_function: str, lattice_shape: Tuple[int, int, int], *args) -> None:
        super().__init__()

        assert all(density_size >= lattice_size for density_size, lattice_size in zip(density_shape, lattice_shape)), f"The density size {density_shape} must be at least as large as the lattice size {lattice_shape} in all dimensions."

        # Minimum and maximum value of density.
        density_range = [0.0, 1.0]
        # Generate the density matrix.
        density_function = self.density_functions[density_function]
        density = density_function(*density_range, *density_shape)

        # Generate the design region.
        x, y, z = np.meshgrid(
            np.arange(lattice_shape[0]) + (density_shape[0] - lattice_shape[0]) // 2,
            np.arange(lattice_shape[1]) + (density_shape[1] - lattice_shape[1]) // 2,
            np.arange(lattice_shape[2]) + (density_shape[2] - lattice_shape[2]) // 2,
        )
        design_region = -1 * np.ones(density.shape)
        design_region[x.flatten(), y.flatten(), z.flatten()] = 1

        # Create the input tensor and normalize it.
        self.inputs = np.concatenate([density[None, None, ...], design_region[None, None, ...]], axis=1)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.inputs -= -0.14252009987831116
        self.inputs /= 0.7857609987258911
    
    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index, ...]

class StrutInferenceDataset(torch.utils.data.Dataset):
    """A dataset for testing the network on custom-generated density. The strut being predicted is fixed at the center of the density matrix."""

    density_functions = {
        'linear': generate_linear_density,
        'sin': generate_sinusoidal_density,
        'exp': generate_exponential_density,
        'random': generate_random_density,
        'stress': load_stress_density,
        'top. opt.': load_topology_optimization_density,
    }

    lattice_functions = {
        'rectangle': generate_rectangular_lattice,
        'circle': generate_circular_lattice,
        'random': generate_random_lattice,
    }

    def __init__(self, density_shape: Tuple[int, int, int], density_function: str, lattice_shape: Tuple[int, int, int], lattice_type: str) -> None:
        super().__init__()

        # Shape of density matrix.
        h, w, d = density_shape
        # Start and end values of density matrix.
        density_range = [0.0, 1.0]

        # Generate a density matrix.
        density_function = self.density_functions[density_function]
        self.density = density_function(*density_range, *density_shape)
        # Normalize density values.
        self.density -= 127.4493
        self.density /= 41.9801
        self.density = torch.tensor(self.density)

        # visualize_input(self.density.numpy(), opacity=1.0)

        # Generate a lattice structure within the volume, as a list containing pairs of node coordinates.
        lattice_function = self.lattice_functions[lattice_type]
        self.indices = lattice_function(tuple(self.density.size()), lattice_shape)
        
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
    pass