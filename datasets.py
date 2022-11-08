"""Dataset classes that load previously cached dataset files."""


import os
import random
import time
from typing import List, Tuple, Union

import torch
# import torch_geometric

from preprocessing import DATASET_FOLDER, read_pickle


class StrutDataset(torch.utils.data.Dataset):
    """Density and node location data as 2-channel 3D tensors and strut diameter data as scalars. The node location channel is a binary array with values {0, 1}.
    
    In this dataset, a single input image corresponds to many struts. Splitting this dataset for training and testing should be done by input image, instead of by strut, in order to ensure that struts corresponding to input images in the training set do not show up in the testing set.

    `count`: Number of input images to include in the dataset. All struts associated with these images are included.
    `p`: Proportion of data to include in the dataset. For example, if 0.1, 10% of the data are randomly sampled. Set a random seed before initializing this dataset to ensure reproducibility.
    `normalize_inputs`: Normalize input values to zero mean and unit variance.
    """

    def __init__(self, count: int=None, p: float=1.0, normalize_inputs: bool=True) -> None:
        super().__init__()
        self.p = p

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs.pickle'))

        if count is not None:
            self.inputs = self.inputs[:count, ...]
            self.outputs = [_ for _ in self.outputs if _[0] < count]
        
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

class CenteredStrutDataset(torch.utils.data.Dataset):
    """Only include 7 struts at the center of the volume per input data."""

    def __init__(self, count: int=None, p: float=1.0, normalize_inputs: bool=True) -> None:
        super().__init__()
        self.p = p

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs.pickle')).float() / 255
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs.pickle'))

        # Remove struts that are not at the center.
        self.outputs = [
            _ for _ in self.outputs if _[1] == (5, 5, 5) and _[2] in [
                # (6, 5, 5),
                # (5, 6, 5),
                # (5, 5, 6),
                # (6, 6, 5),
                # (5, 6, 6),
                (6, 5, 6),
                # (6, 6, 6),
            ]
        ]

        if count is not None:
            self.inputs = self.inputs[:count, ...]
            self.outputs = [_ for _ in self.outputs if _[0] < count]
        
        self.diameters = torch.tensor([output[3] for output in self.outputs])[:, None]
        
        # Normalize input data.
        if normalize_inputs:
            self.input_mean = self.inputs.mean()
            self.input_std = self.inputs.std()

            self.inputs -= self.input_mean
            self.inputs /= self.input_std

        # # Normalize label data.
        # self.diameter_mean = self.diameters.mean().item()
        # self.diameter_std = self.diameters.std().item()
        # self.diameters -= self.diameter_mean
        # self.diameters /= self.diameter_std

        # # Add a second channel for storing node locations. Values should not be normalized.
        # self.inputs = torch.cat([self.inputs, torch.zeros_like(self.inputs)], dim=1)

    def __len__(self) -> int:
        return len(self.outputs)

    def __getitem__(self, index):
        image_index, (x1, y1, z1), (x2, y2, z2), diameter = self.outputs[index]
        return torch.clone(self.inputs[image_index, ...]), torch.clone(self.diameters[index, :])
    
    def outputs_for_images(self, image_indices: List[int]):
        """Return output data indices corresponding to a list of image indices."""
        image_indices = set(image_indices)
        indices = [i for i, output in enumerate(self.outputs) if output[0] in image_indices]
        indices = random.sample(indices, round(self.p * len(indices)))
        return indices


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    dataset = StrutDataset(1000, p=0.01)
    d = dataset.diameters.squeeze().numpy()
    y = d.copy()
    y = ((y - y.mean()) * 3) ** 2.1
    plt.figure()
    plt.plot(np.sort(d))
    plt.plot(np.sort(y))
    # plt.hist(d, bins=25)
    # plt.hist(d_, bins=25)
    plt.show()