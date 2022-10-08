"""Dataset classes that load previously cached dataset files."""


try:
    from google.colab import drive  # type: ignore
except ImportError:
    DATASET_FOLDER = 'Training_Data_10'
else:
    drive.mount('/content/drive')
    DATASET_FOLDER = 'drive/My Drive/Lattice'

import os
import pickle
import time

import torch
import torch_geometric

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

class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, count: int=None) -> None:
        with open(os.path.join(DATASET_FOLDER, 'graphs.pickle'), 'rb') as f:
            self.dataset = pickle.load(f)
        
        if count is not None:
            self.dataset = self.dataset[:count]
        
        # Normalize input data to have zero mean and unit variance.
        inputs = torch.cat([graph.x.flatten() for graph in self.dataset])
        mean, std = inputs.mean(), inputs.std()
        for graph in self.dataset:
            graph.x = graph.x[:, :1]  # Remove coordinate information
            # graph.x -= mean
            # graph.x /= std

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # Return the entire graph. Returning individual attributes (x, edge_index, y) results in incorrect batching.
        return self.dataset[index]