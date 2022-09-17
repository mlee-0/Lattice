"""Dataset classes that load previously cached dataset files."""


import os
import pickle
import time

from torch.utils.data import Dataset


try:
    from google.colab import drive  # type: ignore
except ImportError:
    DATASET_FOLDER = 'Training_Data_50'
else:
    drive.mount('/content/drive')
    DATASET_FOLDER = 'drive/My Drive/Lattice'


class LatticeDataset(Dataset):
    def __init__(self, count: int=None) -> None:
        """
        `count`: The number of data to use, or None to use the entire dataset.
        """

        time_start = time.time()
        
        with open(os.path.join(DATASET_FOLDER, 'inputs.pickle'), 'rb') as f:
            self.inputs = pickle.load(f)
            if count:
                self.inputs = self.inputs[:count, ...]
        with open(os.path.join(DATASET_FOLDER, 'outputs_lattice.pickle'), 'rb') as f:
            self.outputs = pickle.load(f)
            if count:
                self.outputs = self.outputs[:count, ...]

        # Remove density values outside the predefined volume of space.
        with open(os.path.join(DATASET_FOLDER, 'outputs_nodes.pickle'), 'rb') as f:
            nodes = pickle.load(f)
        self.inputs[nodes == 0] = 0
        
        time_end = time.time()
        print(f"Loaded dataset in {round(time_end - time_start)} seconds.")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, ...], self.outputs[index, ...]

class NodeDataset(Dataset):
    def __init__(self) -> None:
        with open(os.path.join(DATASET_FOLDER, 'inputs.pickle'), 'rb') as f:
            self.inputs = pickle.load(f)
        with open(os.path.join(DATASET_FOLDER, 'outputs_nodes.pickle'), 'rb') as f:
            self.outputs = pickle.load(f)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, ...], self.outputs[index, ...]