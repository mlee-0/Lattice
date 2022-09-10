"""Run this file to read and cache dataset files for faster subsequent loading."""


import glob
import os
import pickle
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch_geometric

from datasets import DATASET_FOLDER


# Size of input images (height, width, depth). Images are stacked along the depth dimension.
INPUT_SHAPE = (51, 51, 51)
# Maximum number of struts connected to each node.
STRUTS_PER_NODE = 26


def read_coordinates() -> List[List[int]]:
    """Return a list of lists of X, Y, Z coordinates."""

    filename = 'Input_Coords.txt'
    with open(os.path.join(DATASET_FOLDER, filename), 'r') as f:
        lines = f.readlines()[1:]
    
    coordinates = []
    for line in lines:
        coordinates.append([int(_) for _ in line.split(',')])

    return coordinates

def make_node_numbers() -> np.ndarray:
    """Return a 3D array of node numbers defined in the text file."""
    
    filename = 'Input_Coords.txt'
    with open(os.path.join(DATASET_FOLDER, filename), 'r') as f:
        # Ignore the header line.
        lines = f.readlines()[1:]
    
    size = round(len(lines) ** (1/3))
    numbers = np.zeros([size]*3, dtype=np.int32)

    for number, line in enumerate(lines, 1):
        x, y, z = [int(_) for _ in line.strip().split(',')]
        numbers[x, y, z] = number
    assert np.all(numbers > 0)

    return numbers

def read_struts() -> List[Tuple[int, int]]:
    """Return a list of 2-tuples of the two nodes for each strut."""

    filename = 'Struts.txt'
    with open(os.path.join(DATASET_FOLDER, filename), 'r') as f:
        # Remove the header line.
        lines = f.readlines()[1:]
    
    struts = [tuple(int(number) for number in line.strip().split(',')) for line in lines]

    return struts

def read_inputs(augmentations: int=None) -> np.ndarray:
    """Return input images as a 5D array with shape (number of samples, 1 channel, height, width, depth)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Input_Data')
    folders = next(os.walk(directory))[1]
    folders.sort(key=lambda folder: int(folder.split('_')[1]))

    rotations, axes = cube_rotations(augmentations)

    data_type = np.uint8
    inputs = np.empty((len(folders) * augmentations, 1, *INPUT_SHAPE), dtype=data_type)

    for i, folder in enumerate(folders):
        print(f"Reading images in {folder}...", end='\r')

        files = glob.glob(os.path.join(directory, folder, '*.jpg'))
        files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

        # Concatenate each image.
        for d, file in enumerate(files):
            with Image.open(file) as f:
                inputs[i, 0, ..., d] = np.asarray(f, dtype=data_type)
    
        # Augment the data by rotation.
        for j, (x, y, z) in enumerate(rotations):
            inputs[i + j * len(folders), 0, ...] = np.rot90(np.rot90(np.rot90(inputs[i, 0, ...], x, axes[0]), y, axes[1]), z, axes[2])

    inputs = torch.tensor(inputs)

    return inputs

def read_outputs() -> List[List[Tuple[int, int]]]:
    """Return nonzero-diameter output data as a list of lists of 2-tuples: (strut number, nonzero diameter)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Output_Data')

    files = glob.glob(os.path.join(directory, '*.txt'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    outputs = []

    for i, file in enumerate(files):
        print(f"Reading file {file}...", end='\r')

        with open(file, 'r') as f:
            # Ignore the header line and lines with zero diameters.
            lines = [line for line in f.readlines()[1:] if float(line.strip().split(',')[1]) > 0]
        
        output = []
        for line in lines:
            strut, d = line.strip().split(',')
            output.append((int(strut), float(d)))
        outputs.append(output)

    return outputs

def read_outputs_as_adjacency(augmentations: int=None) -> np.ndarray:
    """Return output data as a 3D array with shape (number of samples, height of the adjacency matrix, width of the adjacency matrix)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Output_Data')

    files = glob.glob(os.path.join(directory, '*.txt'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    # Height of the adjacency matrix, equal to the total number of nodes.
    h = int(np.prod(INPUT_SHAPE))
    # Width of the adjacency matrix, equal to the maximum number of struts per node.
    w = STRUTS_PER_NODE

    struts = read_struts()
    strut_numbers = {strut: number for strut, number in zip(struts, range(1, len(struts)+1))}
    node_numbers = make_node_numbers()

    rotations, axes = cube_rotations(augmentations)
    node_numbers_rotated = [np.rot90(np.rot90(np.rot90(node_numbers, x, axes[0]), y, axes[1]), z, axes[2]) for x, y, z in rotations]

    outputs = np.zeros((len(files) * augmentations, h, w), dtype=np.float32)
    strut_counter = 0

    for i, file in enumerate(files):
        print(f"Reading file {file}...", end='\r')

        with open(file, 'r') as f:
            # Ignore the header line.
            lines = f.readlines()[1:]
        
        for line in lines:
            strut, d = line.strip().split(',')
            strut, d = int(strut), float(d)

            if d > 0:
                strut_counter += 1
                node_1, node_2 = struts[strut - 1]

                for j, node_numbers_ in enumerate(node_numbers_rotated):
                    x1, y1, z1 = np.argwhere(node_numbers_ == node_1)[0, :]
                    x2, y2, z2 = np.argwhere(node_numbers_ == node_2)[0, :]
                    node_1_rotated = node_numbers[x1, y1, z1]
                    node_2_rotated = node_numbers[x2, y2, z2]

                    row = node_1_rotated - 1
                    column = (strut_numbers[(node_1_rotated, node_2_rotated)] - 1) % w

                    outputs[i + j*len(files), row, column] = d

    print(f"\nDensity of adjacency matrix: {strut_counter / (len(files) * h * w)}")

    outputs = torch.tensor(outputs)

    return outputs

def read_outputs_as_nodes() -> np.ndarray:
    """Return output data as a 5D array of 1s indicating nodes with shape (number of samples, 1 channel, height, width, depth)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Output_Data')

    files = glob.glob(os.path.join(directory, '*.txt'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    # Height of the adjacency matrix, equal to the total number of nodes, reduced to remove duplicate nodes.
    h = int(np.prod([INPUT_SHAPE[i] - (3 - 1) for i in range(3)]))
    # Width of the adjacency matrix, equal to the maximum number of struts per node.
    w = STRUTS_PER_NODE

    struts = read_struts()
    nodes = read_coordinates()

    data_type = np.uint8
    outputs = np.zeros([len(files), 1, *INPUT_SHAPE], dtype=data_type)

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

def cube_rotations(count: int=None) -> Tuple[List[Tuple[int, int, int]], Tuple[Tuple[int, int]]]:
    """Return a list of the combinations of unique rotations for a 3D cube, along with the corresponding rotation axes.

    Inputs:
    `count`: Integer [1, 24] representing how many rotations to return, or None to return all possible rotations.

    Outputs:
    `rotations`: List of 3-tuples of the number of rotations to perform in the first, second, and third directions.
    `axes`: Tuple of 2-tuples of the axes defining the plane to rotate in. The first 2-tuple in this tuple corresponds to the first item in each 3-tuple in `rotations`.
    """

    assert 1 <= count <= 24 and type(count) is int

    a = np.arange(3**3).reshape([3]*3)
    rotated = []
    rotations = []
    axes = ((0,1), (1,2), (0,2))
    
    for x in range(4):
        for y in range(4):
            for z in range(4):
                b = np.rot90(np.rot90(np.rot90(a, x, axes=axes[0]), y, axes=axes[1]), z, axes=axes[2])
                for a_ in rotated:
                    if np.all(a_ == b):
                        break
                else:
                    rotated.append(b)
                    rotations.append((x, y, z))
    
    assert rotations[0] == (0, 0, 0)

    # Return the first count rotations only.
    if count is not None:
        rotations = rotations[:count]
    
    return rotations, axes

def augment_inputs(inputs: np.ndarray, augmentations: int=None) -> np.ndarray:
    rotations, axes = cube_rotations(augmentations)

    for kx, ky, kz in rotations:
        rotated = np.rot90(np.rot90(np.rot90(inputs, kx, axes[0]+2), ky, axes[1]+2), kz, axes[2]+2)
        inputs = np.append(inputs, rotated, axis=0)

    return inputs

def augment_outputs(outputs: list, augmentations: int=None):
    rotations, axes = cube_rotations(augmentations)

    node_numbers = make_node_numbers()
    struts = read_struts()

    rotated_outputs = [outputs]

    for kx, ky, kz in rotations:
        rotated_node_numbers = np.rot90(np.rot90(np.rot90(node_numbers, kx, axes[0]), ky, axes[1]), kz, axes[2])

        rotated = []

        for output in outputs:
            for strut, diameter in output:
                node_1, node_2 = struts[strut-1]
                
                x1, y1, z1 = np.argwhere(rotated_node_numbers == node_1)[0, :]
                x2, y2, z2 = np.argwhere(rotated_node_numbers == node_1)[0, :]
                rotated_node_1 = node_numbers[x1, y1, z1]
                rotated_node_2 = node_numbers[x2, y2, z2]

                rotated_strut = struts.index((rotated_node_1, rotated_node_2)) + 1

                rotated.append((rotated_strut, diameter))
        
        rotated_outputs.extend(outputs)

    return rotated_outputs

def convert_dataset_to_graph(inputs: np.ndarray, outputs: list) -> List[torch_geometric.data.Data]:
    """Convert a 5D array of input data and a list of output data to a list of graphs."""

    node_numbers = make_node_numbers()

    graphs = []

    for i in range(len(array.shape[0])):
        indices = np.argwhere(array[i, 0, ...] > 0)

        node_features = torch.empty([len(indices), ...])
        node_coordinates = torch.empty([len(indices), 3])
        edge_indices = torch.empty([2, ...])
        labels = torch.zeros(...)

        for x, y, z in indices:
            node_number = node_numbers[x, y, z]
            node_features[node_number - 1, :] = ...
        
        graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices, y=labels, pos=node_coordinates)
        graphs.append(graph)


if __name__ == "__main__":
    inputs = read_inputs()
    print(inputs.shape)
    # with open('inputs.pickle', 'wb') as f:
    #     pickle.dump(inputs, f)
    
    outputs = read_outputs_as_adjacency(2)
    print(outputs.shape)
    # with open('outputs_lattice.pickle', 'wb') as f:
    #     pickle.dump(outputs, f)
    
    # with open('outputs_lattice.pickle', 'rb') as f:
    #     outputs = pickle.load(f)
    # print('done')
    # # outputs = outputs.to(torch.float16)
    # with open('outputs_lattice_sparse.pickle', 'wb') as f:
    #     pickle.dump(outputs.to_sparse_coo(), f)