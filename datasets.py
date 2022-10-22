"""Dataset classes that load previously cached dataset files."""


try:
    from google.colab import drive  # type: ignore
except ImportError:
    DATASET_FOLDER = 'Training_Data_10'
else:
    drive.mount('/content/drive')
    DATASET_FOLDER = 'drive/My Drive/Lattice'

import os
import random
import time

import torch
# import torch_geometric

from preprocessing import read_pickle


class AdjacencyDataset(torch.utils.data.Dataset):
    def __init__(self, count: int=None) -> None:
        """
        `count`: The number of data to use, or None to use the entire dataset.
        """

        time_start = time.time()
        
        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs_adjacency.pickle')).float()
        
        if count is not None:
            self.inputs = self.inputs[:count, ...]
            self.outputs = self.outputs[:count, ...]

        assert self.inputs.shape[0] == self.outputs.shape[0]

        # Normalize input data to have zero mean and unit variance.
        self.inputs[self.inputs >= 0] -= self.inputs[self.inputs >= 0].mean()
        self.inputs[self.inputs >= 0] /= self.inputs[self.inputs >= 0].std()

        time_end = time.time()
        print(f"Loaded dataset in {round(time_end - time_start)} seconds.")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, ...], self.outputs[index, ...]

class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, count: int=None) -> None:
        super().__init__()

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs_vector.pickle'))

        if count is not None:
            self.inputs = self.inputs[:count, ...]
            self.outputs = self.outputs[:count, ...]
        
        assert self.inputs.shape[0] == self.outputs.shape[0]

        # Normalize input data to have zero mean and unit variance.
        self.inputs[self.inputs >= 0] -= self.inputs[self.inputs >= 0].mean()
        self.inputs[self.inputs >= 0] /= self.inputs[self.inputs >= 0].std()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, ...], self.outputs[index, ...]

# class GraphDataset(torch_geometric.data.Dataset):
#     def __init__(self, count: int=None) -> None:
#         with open(os.path.join(DATASET_FOLDER, 'graphs.pickle'), 'rb') as f:
#             self.dataset = pickle.load(f)
        
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

class LocalDataset(torch.utils.data.Dataset):
    """Stores individual strut diameters and the corresponding 3D density arrays and 3D binary arrays containing node locations."""

    def __init__(self, count: int=None, p: float=1.0) -> None:
        """
        `count`: Number of input images to include in the dataset. All struts associated with these images are included.
        `p`: Proportion of data to include in the dataset. For example, if 0.1, approximately 10% of the data are included.
        """

        super().__init__()
        self.p = p

        self.inputs = read_pickle(os.path.join(DATASET_FOLDER, 'inputs.pickle')).float()
        self.outputs = read_pickle(os.path.join(DATASET_FOLDER, 'outputs_local.pickle'))

        n = len(self.outputs)

        if count is not None:
            self.inputs = self.inputs[:count, ...]
            self.outputs = [_ for _ in self.outputs if _[0] < len(self.inputs)]
        
        self.diameters = torch.tensor([output[3] for output in self.outputs])[:, None]
        
        # Normalize input data to have zero mean and unit variance.
        self.inputs -= self.inputs.mean()
        self.inputs /= self.inputs.std()

        # Add a second channel for storing node locations. Values should not be normalized.
        self.inputs = torch.cat([self.inputs, torch.zeros_like(self.inputs)], dim=1)

    def __len__(self) -> int:
        return len(self.outputs)

    def __getitem__(self, index):
        image_index, (x1, y1, z1), (x2, y2, z2), diameter = self.outputs[index]
        self.inputs[image_index, 1, ...] = 0
        self.inputs[image_index, 1, x1, y1, z1] = 1
        self.inputs[image_index, 1, x2, y2, z2] = 1
        return torch.clone(self.inputs[image_index, ...]), torch.clone(self.diameters[index, :])

    def split_by_input(self, train_split: float=0.8, validate_split: float=0.1, test_split: float=0.1):
        """Return lists of indices for the training/validation/testing datasets, splitting by input image instead of by strut so that the input images in the training set do not appear in the validation or testing sets."""

        random.seed(42)
        image_indices = list(range(len(self.inputs)))
        random.shuffle(image_indices)

        train_size = int(train_split * len(self.inputs))
        validate_size = int(validate_split * len(self.inputs))
        test_size = int(test_split * len(self.inputs))

        train_indices = [i for i, (image_index, *_) in enumerate(self.outputs) if image_index in set(image_indices[:train_size])]
        validate_indices = [i for i, (image_index, *_) in enumerate(self.outputs) if image_index in set(image_indices[train_size:train_size+validate_size])]
        test_indices = [i for i, (image_index, *_) in enumerate(self.outputs) if image_index in set(image_indices[-test_size:])]

        train_indices = random.sample(train_indices, round(self.p * len(train_indices)))
        validate_indices = random.sample(validate_indices, round(self.p * len(validate_indices)))
        test_indices = random.sample(test_indices, round(self.p * len(test_indices)))

        return train_indices, validate_indices, test_indices