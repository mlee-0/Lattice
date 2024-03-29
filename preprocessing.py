"""Run this file to read and cache dataset files for faster subsequent loading."""


import copy
import glob
import math
import os
import pickle
from typing import Any, List, Tuple

import numpy as np
import torch


try:
    from google.colab import drive  # type: ignore
except ImportError:
    DATASET_FOLDER = 'Training_Data_11'
else:
    drive.mount('/content/drive')
    DATASET_FOLDER = 'drive/My Drive/Lattice'


# Size of input images (height, width, depth). Images are stacked along the depth dimension.
INPUT_SHAPE = (11, 11, 11)
# Size of the volume of space around each node inside which struts are formed with other nodes.
STRUT_NEIGHBORHOOD = 3
STRUT_NEIGHBORHOOD_RADIUS = int((STRUT_NEIGHBORHOOD-1) / 2)
# Unit vectors representing the unique, non-parallel struts extending from a node.
DIRECTIONS = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (0, 1, 1), (1, 0, 1), (-1, 1, 0), (0, -1, 1), (-1, 0, 1),
    (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
]


def read_coordinates() -> List[List[int]]:
    """Return a list of lists of X, Y, Z coordinates corresponding to each node."""

    filename = 'Input_Coords.txt'
    with open(os.path.join(DATASET_FOLDER, filename), 'r') as f:
        # Ignore the header line.
        lines = f.readlines()[1:]
    
    coordinates = []
    for line in lines:
        coordinates.append([int(_) for _ in line.split(',')])

    return coordinates

def make_node_numbers(coordinates: list) -> np.ndarray:
    """Return a 3D array of node numbers starting from 1 arranged according to the given list of coordinates."""
    
    size = round(len(coordinates) ** (1/3))
    numbers = np.zeros([size]*3, dtype=np.int32)

    for number, (x, y, z) in enumerate(coordinates, 1):
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

def read_inputs(count: int=None) -> torch.Tensor:
    """Return input images as a 5D tensor with shape (number of data, 1 channel, height, width, depth)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Input_Data')

    files = glob.glob(os.path.join(directory, '*.mat'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    if count is not None:
        files = files[:count]

    data_type = float
    inputs = np.empty((len(files), 1, *INPUT_SHAPE), dtype=data_type)
    for i, file in enumerate(files):
        if i % 10 == 0:
            print(f"Reading file {file}...", end='\r' if i < len(files) - 1 else None)
        inputs[i, 0, ...] = read_mat(file, 'Density')
    
    inputs = torch.tensor(inputs)
    # Fix the order of the dimensions.
    inputs = torch.transpose(inputs, 2, 3)

    return inputs

def read_mat(file: str, key: str) -> np.ndarray:
    """Return the item stored with the given key from a .mat file."""
    
    from scipy.io import loadmat

    # Returns a dictionary.
    data = loadmat(file)
    
    return data[key]

def read_outputs(count: int=None) -> List[List[Tuple[int, int]]]:
    """Return nonzero-diameter output data as a list of lists of 2-tuples: (strut number, nonzero diameter). Duplicate struts are not included, and all strut numbers correspond to node numbers in increasing order (node 1 < node 2)."""
    
    directory = os.path.join(DATASET_FOLDER, 'Output_Data')

    files = glob.glob(os.path.join(directory, '*.txt'))
    files.sort(key=lambda file: int(os.path.basename(file).split('.')[0].split('_')[1]))

    if count is not None:
        files = files[:count]

    struts = read_struts()

    outputs = []

    for i, file in enumerate(files):
        if i % 10 == 0:
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

def get_unique_strut_numbers(struts: list) -> list:
    """Return a list of strut numbers corresponding only to node numbers in increasing order (for example, (1, 2) but not (2, 1))."""

    unique_struts = {tuple(sorted(strut)) for strut in struts}

    strut_numbers = []
    for number, strut in enumerate(struts, 1):
        if strut in unique_struts:
            strut_numbers.append(number)
    
    return strut_numbers

def make_struts(length: int, shape: Tuple[int, int, int]) -> List[Tuple[tuple, tuple]]:
    """Return a list of all unique struts smaller than or equal to the given length, whose coordinates are centered within a `shape`-sized volume. Only returns struts with direction vectors in the first octant (+, +, +).
    
    `length`: The maximum difference in coordinates between two nodes. For example, a length of 3 allows (0, 0, 0) and (3, 3, 3) but not (0, 0, 0) and (4, 4, 4).
    `shape`: The shape of the volume within which the struts are centered.
    """
    
    h, w, d = shape

    # Difference in corresponding coordinates between two nodes. For example, a dx value of 2 results in (0, _, _) and (2, _, _).
    dx = list(range(length+1))
    dy = list(range(length+1))
    dz = list(range(length+1))

    # List of direction vectors pointing from node 1 to node 2. For example, a direction of (0, 1, 2) and having node 1 at the origin (0, 0, 0) results in node 2 being located at (0, 1, 2).
    directions = []
    for x in dx:
        for y in dy:
            for z in dz:
                # Exclude the direction with 0 length.
                if (x, y, z) != (0, 0, 0):
                    directions.append([x, y, z])

    # Shift struts to be centered at within the volume to maximize symmetry within the volume. If a strut is not centered, density values from one side of a strut will be used more than from the opposite side.
    struts = []
    for i, (x, y, z) in enumerate(directions):
        offset_x = (math.ceil(h/2) - 1) - x//2
        offset_y = (math.ceil(w/2) - 1) - y//2
        offset_z = (math.ceil(d/2) - 1) - z//2
        struts.append((
            (0 + offset_x, 0 + offset_y, 0 + offset_z),
            (x + offset_x, y + offset_y, z + offset_z),
        ))
        assert [abs(_1 - _2) for _1, _2 in zip(struts[-1][0], struts[-1][1])] == directions[i]
    
    return struts

def mask_inputs(inputs: torch.Tensor, outputs: list):
    """Replace density values outside the predefined volume of space with some constant value."""

    struts = read_struts()
    coordinates = read_coordinates()
    node_numbers = make_node_numbers(coordinates)

    for i, output in enumerate(outputs):
        mask = mask_of_active_nodes([strut for strut, d in output], struts, node_numbers)
        inputs[i, 0, ~mask] = -1

    return inputs

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

def augment_inputs(inputs: torch.Tensor, augmentations: int=None) -> np.ndarray:
    print(f"Augmenting inputs...")
    rotations, axes = cube_rotations(augmentations)

    axes = tuple((dim_1+2, dim_2+2) for dim_1, dim_2 in axes)
    inputs = np.concatenate(
        [rotate_array(inputs, rotation, axes) for rotation in rotations],
        axis=0,
    )
    inputs = torch.tensor(inputs)

    return inputs

def augment_outputs(outputs: list, augmentations: int=None):
    print(f"Augmenting outputs...")
    rotations, axes = cube_rotations(augmentations)

    coordinates = read_coordinates()
    node_numbers = make_node_numbers(coordinates)
    struts = read_struts()
    strut_indices = {strut: index for index, strut in enumerate(struts)}

    rotated_outputs = copy.deepcopy(outputs)
    for rotation in rotations[1:]:
        rotated_node_numbers = rotate_array(node_numbers, rotation, axes)
        # Dictionary of node indices {node: (x, y, z)} to reduce runtime by avoiding search.
        node_indices = {rotated_node_numbers[x, y, z]: (x, y, z) for x in range(rotated_node_numbers.shape[0]) for y in range(rotated_node_numbers.shape[1]) for z in range(rotated_node_numbers.shape[2])}

        for i, output in enumerate(outputs):
            if i % 10 == 0:
                print(f"Augmenting output {i} of {len(outputs)} for rotation {rotation}...", end='\r')

            rotated_output = [None] * len(output)
            for j, (strut, diameter) in enumerate(output):
                node_1, node_2 = struts[strut - 1]
                
                x1, y1, z1 = node_indices[node_1]
                x2, y2, z2 = node_indices[node_2]
                rotated_node_1 = node_numbers[x1, y1, z1]
                rotated_node_2 = node_numbers[x2, y2, z2]
                # Order node numbers in increasing order.
                rotated_node_1, rotated_node_2 = sorted((rotated_node_1, rotated_node_2))

                rotated_strut = strut_indices[(rotated_node_1, rotated_node_2)] + 1

                rotated_output[j] = (rotated_strut, diameter)
            rotated_outputs.append(rotated_output)

    return rotated_outputs

def convert_outputs_to_struts(outputs: list) -> List[Tuple[int, Tuple[int, int, int], Tuple[int, int, int], float]]:
    """Convert the given output data to a list of tuples containing (input image index, a tuple of (X, Y, Z) coordinates corresponding to node 1 of the strut, a tuple of (X, Y, Z) coordinates corresponding to node 2 of the strut, diameter)."""

    coordinates = read_coordinates()
    struts = read_struts()

    data = []
    for i, output in enumerate(outputs):
        for strut, d in output:
            node_1, node_2 = struts[strut - 1]
            data.append((
                i,
                coordinates[node_1 - 1],
                coordinates[node_2 - 1],
                d,
            ))
    
    return data

def convert_outputs_to_array(outputs: list) -> torch.Tensor:
    """Convert the given output data to a 5D tensor of strut diameters with a number of channels equal to the number of unique struts at each node."""

    coordinates = read_coordinates()
    struts = read_struts()

    array = np.zeros((len(outputs), len(DIRECTIONS), *INPUT_SHAPE))

    for i, output in enumerate(outputs):
        for strut, diameter in output:
            node_1, node_2 = struts[strut - 1]
            (x1, y1, z1), (x2, y2, z2) = coordinates[node_1 - 1], coordinates[node_2 - 1]

            direction_1_2 = (x2 - x1, y2 - y1, z2 - z1)
            direction_2_1 = (x1 - x2, y1 - y2, z1 - z2)
            if direction_1_2 in DIRECTIONS:
                array[i, DIRECTIONS.index(direction_1_2), x1, y1, z1] = diameter
            elif direction_2_1 in DIRECTIONS:
                array[i, DIRECTIONS.index(direction_2_1), x2, y2, z2] = diameter
            else:
                raise Exception(f"Neither {direction_1_2} nor {direction_2_1} could be found.")

    return torch.tensor(array).float()

def mask_of_active_nodes(strut_numbers: list, struts: list, node_numbers: np.ndarray) -> np.ndarray:
    """Return a Boolean array of indicating which nodes are used for the given struts."""
    active_nodes = tuple({node for strut in strut_numbers for node in struts[strut - 1]})
    mask = np.any(
        node_numbers[..., None] == np.array(active_nodes),
        axis=-1,
    )
    return mask

def read_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if type(data) is torch.Tensor:
        print(f"Loaded tensor with size {data.size()} from {path}.")
    else:
        print(f"Loaded {type(data).__name__} with length {len(data):,} from {path}.")

    return data

def write_pickle(data: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {type(data)} with length {len(data)} to {path}.")


if __name__ == "__main__":
    inputs = read_inputs()
    outputs = read_outputs()
    inputs = augment_inputs(inputs)
    outputs = augment_outputs(outputs)
    outputs = convert_outputs_to_array(outputs)
    write_pickle(inputs, 'Training_Data_11/inputs_augmented.pickle')
    write_pickle(outputs, 'Training_Data_11/outputs_array_augmented.pickle')