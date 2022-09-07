"""Run this file to preprocess and cache the dataset."""


import glob
import os
import pickle
import time
from typing import List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, DataLoader


try:
    from google.colab import drive
except ImportError:
    DATASET_FOLDER = 'Training_Data_50'
else:
    drive.mount('/content/drive')
    DATASET_FOLDER = 'drive/My Drive/Lattice'


class LatticeDataset(Dataset):
    def __init__(self) -> None:
        time_start = time.time()
        
        with open(os.path.join(DATASET_FOLDER, 'inputs.pickle'), 'rb') as f:
            self.inputs = pickle.load(f)
        with open(os.path.join(DATASET_FOLDER, 'outputs_lattice.pickle'), 'rb') as f:
            self.outputs = pickle.load(f)

        with open(os.path.join(DATASET_FOLDER, 'outputs_nodes.pickle'), 'rb') as f:
            nodes = pickle.load(f)
        self.inputs[nodes == 0] = 0
        
        time_end = time.time()
        print(f"Loaded dataset in {round(time_end - time_start)} seconds.")
        
        # # Augment the dataset by rotation.
        # self.inputs = torch.cat(
        #     [self.inputs] + [torch.rot90(self.inputs, k, dims) for k in range(1, 4) for dims in [[2, 3]]],
        #     dim=0,
        # )
        # self.outputs = 

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


def read_coordinates() -> List[List[int]]:
    """Return a list of lists of X, Y, Z coordinates."""
    
    with open(os.path.join(DATASET_FOLDER, 'Input_Coords.txt'), 'r') as f:
        lines = f.readlines()[1:]
    
    # nodes = np.zeros((51, 51, 51), dtype=np.int32)
    # for number, line in enumerate(lines, 1):
    #     x, y, z = [int(_) for _ in line.split(',')]
    #     nodes[x, y, z] = number
    # assert np.all(nodes > 0)

    coordinates = []
    for line in lines:
        coordinates.append([int(_) for _ in line.split(',')])

    return coordinates

def read_struts() -> List[List[int]]:
    """Return a list of lists of the two nodes for each strut."""

    with open(os.path.join(DATASET_FOLDER, 'Struts.txt'), 'r') as f:
        # Remove the header line.
        lines = f.readlines()[1:]
    
    struts = [[int(number) for number in line.strip().split(',')] for line in lines]

    # data_type = np.uint32
    # struts = struts.astype(np.uint32)
    # assert struts.max() <= np.iinfo(data_type).max

    return struts

def read_inputs() -> np.ndarray:
    """Return input images as a 5D array with shape (number of samples, 1 channel, height, width, depth)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Input_Data')
    folders = next(os.walk(directory))[1]
    folders.sort(key=lambda folder: int(folder.split('_')[1]))

    data_type = np.uint8
    inputs = np.empty((len(folders), 1, 51, 51, 51), dtype=data_type)

    for i, folder in enumerate(folders):
        print(f"Reading images in {folder}...", end='\r')

        files = glob.glob(os.path.join(directory, folder, '*.jpg'))
        files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

        for d, file in enumerate(files):
            with Image.open(file) as f:
                inputs[i, 0, ..., d] = np.asarray(f, dtype=data_type)
    
    inputs = torch.tensor(inputs)

    return inputs

def read_outputs() -> np.ndarray:
    """Return output data as a 3D array with shape (number of samples, height of the adjacency matrix, width of the adjacency matrix)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Output_Data')

    files = glob.glob(os.path.join(directory, '*.txt'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    # Dimensions of the adjacency matrix.
    h = (51 - (3-1)) ** 3  # Total number of nodes, reduced to remove duplicate nodes
    w = (3**3) - 1  # Total number of struts per node in a 3x3x3 neighborhood

    outputs = torch.zeros((len(files), h, w), dtype=torch.float32)

    struts = read_struts()

    strut_counter = 0

    for i, file in enumerate(files):
        print(f"Reading file {file}...", end='\r')

        with open(file, 'r') as f:
            # Ignore the header line.
            _ = f.readline()
            # Read all lines except the lines at the bottom containing duplicate struts.
            for line in range(1, h*w + 1):
                strut, d = f.readline().strip().split(',')
                strut, d = int(strut), float(d)

                if d > 0:
                    strut_counter += 1
                    node_1, node_2 = struts[strut - 1]
                    column = (strut - 1) % w

                    outputs[i, node_1-1, column] = d
    
    print(f"\nDensity of adjacency matrix: {strut_counter / (len(files) * h * w)}")

    return outputs

def read_outputs_as_nodes() -> np.ndarray:
    """Return output data as a 5D array of 1s indicating nodes with shape (number of samples, 1 channel, height, width, depth)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Output_Data')

    files = glob.glob(os.path.join(directory, '*.txt'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    # Dimensions of the adjacency matrix.
    h = (51 - (3-1)) ** 3  # Total number of nodes, reduced to remove duplicate nodes
    w = (3**3) - 1  # Total number of struts per node in a 3x3x3 neighborhood

    struts = read_struts()
    nodes = read_coordinates()

    data_type = np.uint8
    outputs = np.zeros([len(files), 1, 51, 51, 51], dtype=data_type)

    for i, file in enumerate(files):
        print(f"Reading file {file}...", end='\r')

        with open(file, 'r') as f:
            # Ignore the header line.
            _ = f.readline()
            # Read all lines except the lines at the bottom containing duplicate struts.
            for line in range(1, h*w + 1):
                strut, d = f.readline().strip().split(',')
                strut, d = int(strut), float(d)

                if d > 0:
                    for node in struts[strut - 1]:
                        x, y, z = nodes[node]
                        outputs[i, 0, x, y, z] = 1
    
    outputs = torch.tensor(outputs)

    return outputs


if __name__ == "__main__":
    # inputs = read_inputs()
    # with open('inputs.pickle', 'wb') as f:
    #     pickle.dump(inputs, f)
    
    # outputs = read_outputs_as_nodes()
    # with open('outputs_nodes.pickle', 'wb') as f:
    #     pickle.dump(outputs, f)
    
    # outputs = read_outputs()
    # with open('outputs_lattice.pickle', 'wb') as f:
    #     pickle.dump(outputs, f)
    
    with open('outputs_lattice.pickle', 'rb') as f:
        outputs = pickle.load(f)
    print('done')
    # outputs = outputs.to(torch.float16)
    with open('outputs_lattice_sparse.pickle', 'wb') as f:
        pickle.dump(outputs.to_sparse_coo(), f)