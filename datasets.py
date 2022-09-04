"""Run this file to preprocess and cache the dataset."""


import glob
import os
import pickle
from typing import List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, DataLoader


class LatticeDataset(Dataset):
    def __init__(self, folder: str) -> None:
        with open(os.path.join(folder, 'cache.pickle'), 'rb') as f:
            self.inputs, self.outputs = pickle.load(f)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, ...], self.outputs[index, ...]


def read_struts() -> List[List[int]]:
    with open('Training_Data_50/Struts.txt', 'r') as f:
        # Remove the header line.
        lines = f.readlines()[1:]
    
    struts = [[int(number) for number in line.strip().split(',')] for line in lines]

    # data_type = np.uint32
    # struts = struts.astype(np.uint32)
    # assert struts.max() <= np.iinfo(data_type).max

    return struts

def read_inputs() -> np.ndarray:
    """Return input images as a single 5D array with shape (number of samples, 1 channel, height, width, depth)."""
    
    directory = 'Training_Data_50/Input_Data'
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
                inputs[i, 0, ..., d] = (image:=np.asarray(f, dtype=data_type))
    
    inputs = torch.tensor(inputs)

    return inputs

def read_outputs() -> np.ndarray:
    """Return output data as a single 3D array with shape (number of samples, height of the adjacency matrix, width of the adjacency matrix)."""
    
    directory = 'Training_Data_50/Output_Data'

    files = glob.glob(os.path.join(directory, '*.txt'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    # Dimensions of the adjacency matrix.
    h = (51 - (3-1)) ** 3  # Total number of nodes, reduced to remove duplicate nodes
    w = (3**3) - 1  # Total number of struts per node in a 3x3x3 neighborhood

    indices = []
    values = []

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

                if d != 0:
                    strut_counter += 1
                    node_1, node_2 = struts[strut - 1]
                    column = (strut - 1) % w

                    indices.append([i, node_1-1, column])
                    values.append(d)
    
    print(f"\nDensity of adjacency matrix: {strut_counter / (len(files) * h * w)}")

    outputs = torch.sparse_coo_tensor(np.array(indices).transpose(), values, (len(files), h, w), dtype=torch.float32)

    return outputs

inputs = read_inputs()
with open('inputs.pickle', 'wb') as f:
    pickle.dump(inputs, f)
outputs = read_outputs()
with open('outputs.pickle', 'wb') as f:
    pickle.dump(outputs, f)


# if __name__ == "__main__":
#     file = "Training_Data_50/Struts.txt"
#     with open(file, 'r') as f:
#         for i, line in enumerate(f.readlines()):
#             if '116420,119021' in line:
#                 print(i)

#         # # print(lines[2957942])
#         # print(lines[3094521])

#     # folder = "Training_Data_50/Output_Data"
#     # files = glob.glob(os.path.join(folder, "*.txt"))
#     # files.sort()
#     # for file in files[:1]:
#     #     print(file, end='\r')
#     #     with open(file, 'r') as f:
#     #         lines = f.readlines()[1:]
#     #         lines = lines[(49**3 * 26):]
#     #         for line in lines:
#     #             if '.' in line.strip().split(',')[1]:
#     #                 print(line)
#     #         # if any('.' in line.strip().split(',')[1] for line in lines):
#     #         #     print(f"redundant diameters found in {file}")