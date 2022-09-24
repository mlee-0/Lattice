"""Dataset classes that load previously cached dataset files."""


import os
import pickle
import time

import torch
import torch_geometric


try:
    from google.colab import drive  # type: ignore
except ImportError:
    DATASET_FOLDER = 'Training_Data_50'
else:
    drive.mount('/content/drive')
    DATASET_FOLDER = 'drive/My Drive/Lattice'


class CnnDataset(torch.utils.data.Dataset):
    def __init__(self, count: int=None) -> None:
        """
        `count`: The number of data to use, or None to use the entire dataset.
        """

        time_start = time.time()
        
        with open(os.path.join(DATASET_FOLDER, 'inputs.pickle'), 'rb') as f:
            self.inputs = pickle.load(f).float()
        with open(os.path.join(DATASET_FOLDER, 'outputs_adjacency.pickle'), 'rb') as f:
            self.outputs = pickle.load(f).float()
        
        if count is not None:
            self.inputs = self.inputs[:count, ...]
            self.outputs = self.outputs[:count, ...]

        assert self.inputs.shape[0] == self.outputs.shape[0]

        # # Remove density values outside the predefined volume of space.
        # with open(os.path.join(DATASET_FOLDER, 'outputs_nodes.pickle'), 'rb') as f:
        #     nodes = pickle.load(f)
        # self.inputs[nodes == 0] = 0
        
        time_end = time.time()
        print(f"Loaded dataset in {round(time_end - time_start)} seconds.")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, ...], self.outputs[index, ...]

class GnnDataset(torch_geometric.data.Dataset):
    def __init__(self, count: int=None) -> None:
        with open(os.path.join(DATASET_FOLDER, 'outputs_graph.pickle'), 'rb') as f:
            self.dataset = pickle.load(f)
        
        if count is not None:
            self.dataset = self.dataset[:count]
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # Returning x, edge_index, y individually results in incorrect batching.
        return self.dataset[index]