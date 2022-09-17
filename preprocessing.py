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
    """Return a 3D array of node numbers starting from 1 defined in the text file."""
    
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

def read_inputs() -> np.ndarray:
    """Return input images as a 5D array with shape (number of samples, 1 channel, height, width, depth)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Input_Data')
    folders = next(os.walk(directory))[1]
    folders.sort(key=lambda folder: int(folder.split('_')[1]))

    data_type = np.uint8
    inputs = np.empty((len(folders), 1, *INPUT_SHAPE), dtype=data_type)

    for i, folder in enumerate(folders):
        print(f"Reading images in {folder}...", end='\r' if i < len(folders) - 1 else None)

        files = glob.glob(os.path.join(directory, folder, '*.jpg'))
        files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

        # Concatenate each image.
        for d, file in enumerate(files):
            with Image.open(file) as f:
                inputs[i, 0, ..., d] = np.asarray(f, dtype=data_type)

    inputs = torch.tensor(inputs)

    return inputs

def read_outputs() -> List[List[Tuple[int, int]]]:
    """Return nonzero-diameter output data as a list of lists of 2-tuples: (strut number, nonzero diameter). Duplicate struts are not included."""
    
    directory = os.path.join(DATASET_FOLDER, 'Output_Data')

    files = glob.glob(os.path.join(directory, '*.txt'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    struts = read_struts()

    outputs = []

    for i, file in enumerate(files):
        print(f"Reading file {file}...", end='\r' if i < len(files) - 1 else None)

        with open(file, 'r') as f:
            # Ignore the header line and lines with zero diameters.
            lines = [line for line in f.readlines()[1:] if float(line.strip().split(',')[1]) > 0]
        
        output = []
        unique_nodes = set()
        for line in lines:
            strut, d = line.strip().split(',')
            strut, d = int(strut), float(d)
            nodes = tuple(sorted(struts[strut - 1]))
            if nodes not in unique_nodes:
                output.append((strut, d))
            unique_nodes.add(nodes)
        outputs.append(output)

    return outputs

def convert_outputs_to_adjacency(outputs: list, augmentations: int=None) -> np.ndarray:
    """Convert the given output data to a 3D array with shape (number of samples, height of the adjacency matrix, width of the adjacency matrix)."""

    # Number of data.
    n = len(outputs)

    # Height of the adjacency matrix, equal to the total number of nodes.
    h = int(np.prod(INPUT_SHAPE))
    # Width of the adjacency matrix, equal to the maximum number of struts per node.
    w = STRUTS_PER_NODE

    struts = read_struts()
    strut_numbers = {strut: number for strut, number in zip(struts, range(1, len(struts)+1))}
    node_numbers = make_node_numbers()

    rotations, axes = cube_rotations(augmentations)
    node_numbers_rotated = [rotate_array(node_numbers, rotation, axes) for rotation in rotations]

    outputs = np.zeros((n * augmentations, h, w), dtype=np.float32)
    strut_counter = 0

    for i, output in enumerate(outputs):
        print(f"Processing output {i} of {n}...", end='\r')

        for strut, d in output:
            strut_counter += 1
            node_1, node_2 = struts[strut - 1]

            for j, node_numbers_ in enumerate(node_numbers_rotated):
                x1, y1, z1 = np.argwhere(node_numbers_ == node_1)[0, :]
                x2, y2, z2 = np.argwhere(node_numbers_ == node_2)[0, :]
                node_1_rotated = node_numbers[x1, y1, z1]
                node_2_rotated = node_numbers[x2, y2, z2]

                row = node_1_rotated - 1
                column = (strut_numbers[(node_1_rotated, node_2_rotated)] - 1) % w

                outputs[i + j*n, row, column] = d

    print(f"\nDensity of adjacency matrix: {strut_counter / (n * h * w)}")

    outputs = torch.tensor(outputs)

    return outputs

def cube_rotations(count: int=None) -> Tuple[List[Tuple[int, int, int]], Tuple[Tuple[int, int]]]:
    """Return a list of the unique rotations for a 3D cube, along with the corresponding rotation axes.

    Inputs:
    `count`: Integer [1, 24] representing how many rotations to return, or None to return all possible rotations.

    Outputs:
    `rotations`: List of 3-tuples of the number of rotations to perform in the first, second, and third directions.
    `axes`: Tuple of 2-tuples of the axes defining the plane to rotate in. The first 2-tuple in this tuple corresponds to the first item in each 3-tuple in `rotations`.
    """

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
        assert 1 <= count <= 24
        rotations = rotations[:count]
    
    return rotations, axes

def rotate_array(array: np.ndarray, rotation: tuple, axes: tuple) -> np.ndarray:
    """Rotate an array about the given axes the corresponding number of times."""
    assert len(rotation) == len(axes)

    for k, axis in zip(rotation, axes):
        array = np.rot90(array, k, axis)

    return array

def augment_inputs(inputs: np.ndarray, augmentations: int=None) -> np.ndarray:
    rotations, axes = cube_rotations(augmentations)

    axes = tuple((dim_1+2, dim_2+2) for dim_1, dim_2 in axes)
    inputs = np.concatenate(
        [rotate_array(inputs, rotation, axes) for rotation in rotations],
        axis=0,
    )

    return inputs

def augment_outputs(outputs: list, augmentations: int=None):
    rotations, axes = cube_rotations(augmentations)

    node_numbers = make_node_numbers()
    struts = read_struts()

    rotated_outputs = [outputs]

    for rotation in rotations:
        rotated_node_numbers = rotate_array(node_numbers, rotation, axes)

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

    assert inputs.shape[0] == len(outputs), f"Number of inputs {inputs.shape[0]} and number of outputs {len(outputs)} do not match."
    n = len(outputs)

    node_numbers = make_node_numbers()
    struts = read_struts()

    graphs = []

    for i in range(n):
        mask = mask_of_active_nodes([_[0] for _ in outputs[i]], struts, node_numbers)
        indices = np.argwhere(mask)

        number_nodes = indices.shape[0]
        number_total_nodes = node_numbers.size

        # Node feature matrix with shape (number of nodes, number of features per node). Includes all possible nodes, not just the nodes with nonzero values, to avoid having to renumber nodes.
        node_features = torch.empty([number_total_nodes, 1])
        # Node coordinate matrix with shape (number of nodes, number of features per node).
        node_coordinates = torch.empty([number_total_nodes, 3])
        # List of edges as 2-tuples (node 1, node 2). Struts are formed within a 3x3x3 neighborhood.
        edge_index = set()
        
        for x, y, z in indices:
            node = node_numbers[x, y, z]
            # Insert density of each node.
            node_features[node-1, 0] = inputs[i, 0, x, y, z]
            # Insert coordinates of each node.
            node_coordinates[node-1, 0] = x
            node_coordinates[node-1, 1] = y
            node_coordinates[node-1, 2] = z

            # Insert edges for all valid struts.
            r = 1
            neighborhood = node_numbers[
                max(0, x-r):min(INPUT_SHAPE[0], x+r+1),
                max(0, y-r):min(INPUT_SHAPE[1], y+r+1),
                max(0, z-r):min(INPUT_SHAPE[2], z+r+1),
            ]
            for x_, y_, z_ in np.argwhere(neighborhood):
                node_2 = neighborhood[x_, y_, z_]
                if node != node_2:
                    edge_index.add(tuple(sorted((node, node_2))))
        
        # Each strut must be represented by two separate edges in opposite directions to make the graph undirected (both (1, 2) and (2, 1)).
        edge_index = list(edge_index)
        edge_index.extend([strut[::-1] for strut in edge_index])
        # Dictionary of (strut, indices) pairs to reduce runtime by avoiding list search.
        edge_index_mapping = {strut: index for index, strut in enumerate(edge_index)}

        # Edge labels with shape (number of edges, 1).
        labels = torch.zeros([len(edge_index)//2, 1])
        for strut, diameter in outputs[i]:
            edge = edge_index_mapping[struts[strut - 1]]
            assert edge < (len(edge_index) / 2)
            labels[edge, 0] = diameter

        # Graph connectivity matrix transposed into shape (2, number of edges), where each column contains the two nodes that form an edge.
        edge_index = np.array(edge_index).transpose()

        graph = torch_geometric.data.Data(x=node_features, edge_index=edge_index, y=labels, pos=node_coordinates)
        graphs.append(graph)
    
    return graphs

def mask_of_active_nodes(strut_numbers: list, struts: list, node_numbers: np.ndarray) -> np.ndarray:
    """Return a Boolean array of indicating which nodes are used for the given struts."""
    active_nodes = tuple({node for strut in strut_numbers for node in struts[strut - 1]})
    mask = np.any(
        node_numbers[..., None] == np.array(active_nodes),
        axis=-1,
    )
    return mask


if __name__ == "__main__":
    inputs = read_inputs()
    # # with open('inputs.pickle', 'wb') as f:
    # #     pickle.dump(inputs, f)
    
    outputs = read_outputs()
    # outputs = convert_outputs_to_adjacency(outputs)
    outputs = convert_dataset_to_graph(inputs[:1, ...], outputs)

    # # Visualize the neighborhood of a node
    # struts = read_struts()
    # node_numbers = make_node_numbers()
    # # struts = {tuple(sorted(strut)) for strut in struts}
    # struts = [strut for strut in struts if strut[0] == 50000]
    # nodes = set()
    # for strut in struts:
    #     nodes.add(strut[0])
    #     nodes.add(strut[1])
    # array = np.zeros_like(node_numbers)
    # for node in nodes:
    #     array[node_numbers == node] = 1
    # # array = array[:10, :10, :10]
    # from visualization import plot_nodes
    # plot_nodes(array, 0.5)