"""
Functions for generating figures and visualizations.
"""


import matplotlib.pyplot as plt
import vtk

from preprocessing import DIRECTIONS, read_pickle, cube_rotations
from visualization import *


def visualize_density():
    """Show a relative density matrix."""

    data = read_pickle('Training_Data_11/inputs_augmented.pickle').numpy()
    actor = make_actor_density(data[0, 0, ...] * 255, hide_zeros=False)
    visualize_actors(actor, gui=True)

def visualize_design_region():
    """Show a design region matrix."""

    data = read_pickle('Training_Data_11/outputs_array_augmented.pickle').numpy()
    actor = make_actor_density((data[0, 0, ...] > 0) * 255, use_lighting=True, hide_zeros=True)
    visualize_actors(actor, gui=True)

def visualize_design_region_triangulation():
    """Show a design region as a triangulated surface."""

    coordinates = [
        [5.18751889795213,3.15247116832202,0.214900082320982],
        [4.16822862672221,1.54186308339116,0.117568878880779],
        [1.08547556984511,3.88423229604318,3.98796665192688],
        [6.31427675090401,3.05439848198947,2.09695674834458],
        [1.26733568056866,8.2117581307302,5.87914740177718],
        [5.80643663513365,2.79882346584207,8.22721895086327],
        [1.84500375545941,8.51453947760294,7.49075982358463],
        [4.94911090203233,1.95285155964464,9.29144914732334],
        [3.35440618316172,3.55002466794182,5.24545534321685],
        [2.068487698259,3.45975581098889,5.38645416634196],
        [1.05165521104586,5.85631351305979,6.67294499380659],
        [1.53632517968856,4.56250795406776,8.71261196741571],
        [8.73329549947576,0.384487116383723,1.81324929181593],
        [7.24371498334733,3.80015594346897,6.99106624590959],
        [0.0358417601612571,5.46487045006288,1.88910203261924],
        [8.10415238012061,0.682580666253524,1.99816212424685],
        [2.45622407967823,2.13226465005379,7.70965949106791],
        [8.85936548815393,7.71897567107188,5.29896188205266],
        [8.39280649475202,6.30379369335341,0.0471463719527176],
        [3.55567385189081,8.08721360720695,9.55361050072058],
    ]
    triangulation = [
        [17,12,10],
        [9,17,10],
        [16,19,4],
        [1,4,19],
        [4,14,6],
        [9,4,6],
        [9,6,17],
        [7,9,5],
        [11,7,5],
        [11,10,12],
        [3,10,11],
        [7,12,20],
        [11,12,7],
        [9,7,20],
        [3,5,9],
        [11,5,15],
        [3,15,5],
        [3,11,15],
        [8,12,17],
        [6,8,17],
        [18,14,4],
        [9,18,4],
        [9,14,18],
        [3,2,10],
        [9,10,2],
        [2,16,4],
        [1,16,2],
        [20,12,8],
        [6,20,8],
        [9,1,3],
        [20,6,14],
        [9,20,14],
        [15,1,2],
        [3,15,2],
        [3,1,15],
        [9,2,4],
        [9,4,1],
        [13,19,16],
        [1,13,16],
        [1,19,13],
    ]

    datas = vtk.vtkAppendPolyData()
    for i, j, k in triangulation:
        points = vtk.vtkPoints()
        points.InsertNextPoint(coordinates[i-1])
        points.InsertNextPoint(coordinates[j-1])
        points.InsertNextPoint(coordinates[k-1])

        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, 0)
        triangle.GetPointIds().SetId(1, 1)
        triangle.GetPointIds().SetId(2, 2)

        triangles = vtk.vtkCellArray()
        triangles.InsertNextCell(triangle)

        data = vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetPolys(triangles)
        datas.AddInputData(data)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(datas.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLighting(True)

    visualize_actors(actor, gui=True)

def visualize_output():
    """Show a lattice structure."""

    data = read_pickle('Training_Data_11/outputs_array_augmented.pickle').numpy()
    data = data[0, ...]
    actor = make_actor_lattice(*convert_array_to_lattice(data), resolution=100)
    visualize_actors(actor, gui=True)

def visualize_output_by_channel(channel: int):
    """Visualization of only the struts for the corresponding channel."""

    data = read_pickle('Training_Data_11/outputs_array_augmented.pickle').numpy()
    data = data[0, ...]
    data[:channel, ...] = 0
    data[channel+1:, ...] = 0
    actor = make_actor_lattice(*convert_array_to_lattice(data), resolution=100)
    visualize_actors(actor, gui=True)

def plot_output_by_channel():
    """Show each channel of the output matrix and save as files."""

    data = read_pickle('Training_Data_11/outputs_array_augmented.pickle').numpy()
    # Transpose and flip the array to match the coordinate system of the VTK visualization.
    data = np.transpose(data, (0, 1, 2, 4, 3))
    data = data[..., :, ::-1, :]

    for i in range(data.shape[1]):
        figure = plt.figure(figsize=(2,2))
        axis = figure.add_subplot(1, 1, 1, projection='3d')
        axis.voxels(data[0, i, ...] > 0, facecolors=np.repeat(data[0, i, ..., None], 3, axis=-1))
        # axis.axis(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
        plt.savefig(f'matrix_{i+1}.png')

    plt.show()

def visualize_struts():
    """Show the 26 struts extending from each node."""

    actor = make_actor_lattice(
        locations_1=[(0, 0, 0)]*26,
        locations_2=[(sign*x, sign*y, sign*z) for x, y, z in DIRECTIONS for sign in (-1, +1)],
        diameters=[0.25]*26,
        resolution=100,
    )
    visualize_actors(actor, gui=True)

def visualize_unique_struts(channel: int=None):
    """Show the 13 unique struts extending from each node. If `channel` is provided, use a different color for the strut for the corresponding channel."""

    actor = make_actor_lattice(
        locations_1=[(0, 0, 0)]*len(DIRECTIONS),
        locations_2=DIRECTIONS,
        diameters=[0.25]*len(DIRECTIONS),
        resolution=100,
    )

    if channel is not None:
        actor_highlight = make_actor_lattice(
            locations_1=[(0, 0, 0)],
            locations_2=[DIRECTIONS[channel]],
            diameters=[0.25],
            resolution=100,
        )
        actor_highlight.GetProperty().SetColor(0/255, 191/255, 96/255)
        visualize_actors(actor, actor_highlight, gui=False, screenshot_filename=f'strut_{channel+1}.png')
    else:
        visualize_actors(actor, gui=True)

def plot_data_augmentation():
    """Show a relative density matrix rotated 24 ways."""

    data = read_pickle('Training_Data_11/inputs_augmented.pickle').numpy()

    rotations, _ = cube_rotations()

    figure = plt.figure(figsize=(5, 5))
    for i, index in enumerate(range(0, len(data), 1000)):
        axis = figure.add_subplot(6, 4, i+1, projection='3d')
        axis.voxels(data[index, 0, ...] * 255, facecolors=np.repeat(data[index, 0, ..., None], 3, axis=-1))
        axis.axis(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
        axis.set_title(f'{tuple(_*90 for _ in rotations[i])}')
    
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.show()

def plot_clipping():
    """Plot showing how LatticeNet clips output data to [0, 100]."""

    x = np.linspace(-100, 200, 100)
    plt.figure()
    plt.plot(x, np.clip(x, 0, 100))
    plt.xticks([0, 100])
    plt.yticks([0, 100])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_errors_by_channel():
    """Show large errors found in the 13th channel of output."""

    mae_before = [0.0035, 0.0034, 0.0033, 0.0032, 0.0027, 0.0032, 0.0022, 0.0030, 0.0023, 0.0029, 0.0018, 0.0030, 0.0294]
    mae_nonzero_before = [0.0449, 0.0426, 0.0420, 0.0491, 0.0407, 0.0489, 0.0339, 0.0474, 0.0360, 0.0501, 0.0315, 0.0538, 0.5504]

    mae_after = [0.0024543037, 0.0028024123, 0.0025359404, 0.0025793596, 0.0027589872, 0.0025958207, 0.0023535574, 0.001957662, 0.002009013, 0.0028329478, 0.0022655376, 0.002060359, 0.0016395957]
    mae_nonzero_after = [0.030800002, 0.033750065, 0.029547635, 0.038491372, 0.039132852, 0.037395414, 0.03506133, 0.02897873, 0.028993262, 0.04718298, 0.03725247, 0.035770904, 0.028654413]

    for mae, mae_nonzero in ([mae_before, mae_nonzero_before], [mae_after, mae_nonzero_after]):
        plt.figure(figsize=(5, 2.5))
        plt.plot(range(1, 13+1), mae, '.-', label='All')
        plt.plot(range(1, 13+1), mae_nonzero, '.-', label='Nonzero')
        plt.xlabel('Channel')
        plt.ylabel('MAE')
        plt.xticks(range(1, 13+1))
        plt.ylim([0, 0.6])
        plt.legend()

    plt.show()

if __name__ == '__main__':
    pass