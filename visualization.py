import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # type: ignore

from datasets import *
from gui import InferenceWindow
from preprocessing import *


def plot_nodes(array: np.ndarray, opacity: float=1.0) -> None:
    """Show a 3D plot of node locations."""

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.voxels(
        filled=(array > 0),
        facecolors=(1, 1, 1, opacity),
        linewidth=0.25,
        edgecolors=(1, 1, 1),
    )
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    # ax.set(xlim=axis_limits, ylim=axis_limits, zlim=axis_limits)
    plt.show()

def convert_output_to_lattice(output: list) -> Tuple[list, list, list]:
    """Convert a list of tuples (strut number, diameter) into a tuple of coordinates and diameters."""

    coordinates = read_coordinates()
    struts = read_struts()

    coordinates_1, coordinates_2, diameters = [], [], []

    for strut, diameter in output:
        node_1, node_2 = struts[strut - 1]
        coordinates_1.append(tuple(coordinates[node_1 - 1]))
        coordinates_2.append(tuple(coordinates[node_2 - 1]))
        diameters.append(diameter)
    
    return coordinates_1, coordinates_2, diameters

def convert_array_to_lattice(array: np.ndarray) -> Tuple[list, list, list]:
    """Convert a 4D array into a tuple of coordinates and diameters."""

    indices = np.argwhere(array)

    coordinates_1, coordinates_2, diameters = [], [], []

    for channel, x, y, z in indices:
        dx, dy, dz = DIRECTIONS[channel]
        coordinates_1.append([x, y, z])
        coordinates_2.append([x + dx, y + dy, z + dz])
        diameters.append(array[channel, x, y, z])

    return coordinates_1, coordinates_2, diameters

def convert_adjacency_to_lattice(array: np.ndarray) -> Tuple[list, list, list]:
    """Convert a 2D adjacency matrix into a tuple of coordinates and diameters."""

    coordinates = read_coordinates()
    node_numbers = make_node_numbers(coordinates)

    coordinates_1, coordinates_2, diameters = [], [], []
    unique_struts = set()

    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            diameter = array[row, column]

            if diameter > 0:
                x1, y1, z1 = np.unravel_index(row, shape=(11,)*3)
                if column >= 13:
                    column += 1
                x2, y2, z2 = np.unravel_index(column, shape=(3,)*3) + np.array([x1-1, y1-1, z1-1])

                # Skip nodes near the edges that may result in out-of-bounds indices.
                try:
                    node_1, node_2 = node_numbers[x1, y1, z1], node_numbers[x2, y2, z2]
                except IndexError:
                    continue
                
                # Skip duplicate struts.
                strut = tuple(sorted((node_1, node_2)))
                if strut not in unique_struts:
                    unique_struts.add(strut)

                    coordinates_1.append(tuple(coordinates[node_1 - 1]))
                    coordinates_2.append(tuple(coordinates[node_2 - 1]))
                    diameters.append(diameter)
    
    return coordinates_1, coordinates_2, diameters

def convert_vector_to_lattice(vector: np.ndarray) -> Tuple[list, list, list]:
    """Convert a 1D array into a tuple of coordinates and diameters."""
    
    struts = read_struts()
    coordinates = read_coordinates()
    strut_numbers = get_unique_strut_numbers(struts)

    coordinates_1, coordinates_2, diameters = [], [], []

    for i, diameter in enumerate(vector):
        if diameter > 0:
            node_1, node_2 = struts[strut_numbers[i] - 1]
            assert node_1 < node_2

            coordinates_1.append(tuple(coordinates[node_1 - 1]))
            coordinates_2.append(tuple(coordinates[node_2 - 1]))
            diameters.append(diameter)
    
    return coordinates_1, coordinates_2, diameters

def convert_graph_to_lattice(graph) -> Tuple[list, list, list]:
    """Convert a graph into a tuple of coordinates and diameters."""

    coordinates = read_coordinates()

    coordinates_1, coordinates_2, diameters = [], [], []

    for i in range(graph.edge_index.size(1) // 2):
        node_1, node_2 = graph.edge_index[:, i]
        coordinates_1.append(tuple(coordinates[node_1]))
        coordinates_2.append(tuple(coordinates[node_2]))
        diameters.append(graph.y[i])

    return coordinates_1, coordinates_2, diameters

def make_actor_density(array: np.ndarray, opacity: float=1.0, length: float=1.0, use_lighting: bool=False, hide_zeros: bool=False):
    """Return an actor of a voxel model.
    
    `array`: A 3D (grayscale) or 4D (color) array with values in [0, 255]. If 4D, the color dimension must have values in [0, 1] and must be the fourth dimension, with shape (h, w, d, 3).
    `opacity`: The opacity of the actor, with a value in [0, 1].
    `length`: The size of each voxel.
    `use_lighting`: True to enable lighting on the actor.
    `hide_zeros`: Hide voxels with values of 0.
    """

    array = np.array(array)
    
    points = vtk.vtkPoints()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("colors")
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            for z in range(array.shape[2]):
                if array.ndim == 3:
                    if not hide_zeros or array[x, y, z] > 0:
                        points.InsertNextPoint(x, y, z)
                        colors.InsertNextTuple([array[x, y, z]] * 3)
                else:
                    if not hide_zeros or (array[x, y, z, :] == 0).all():
                        points.InsertNextPoint(x, y, z)
                        colors.InsertNextTuple(list(array[x, y, z, :]))

    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.GetPointData().AddArray(colors)

    glyph = vtk.vtkCubeSource()
    glyph.SetXLength(length)
    glyph.SetYLength(length)
    glyph.SetZLength(length)
    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetSourceConnection(glyph.GetOutputPort())
    mapper.SetInputData(data)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("colors")
    mapper.Update()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(1)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetLighting(use_lighting)

    return actor

def make_actor_lattice(locations_1: List[Tuple[float, float, float]], locations_2: List[Tuple[float, float, float]], diameters: List[float], resolution: int=5, show_bounding_box: bool=False, translation: Tuple[float, float, float]=None):
    """Return an actor of a lattice defined as a list of node 1 coordinates, a list of node 2 coordinates, and a list of diameters. All lists must be the same length.
    
    `resolution`: The number of sides on each tube.
    `show_bounding_box`: Show an outline of the bounding box of the volume.
    `translation`: The (X, Y, Z) coordinates by which to shift the entire lattice.
    """

    assert len(locations_1) == len(locations_2) == len(diameters)

    data = vtk.vtkAppendPolyData()

    volume = 0
    for i, ((x1, y1, z1), (x2, y2, z2), diameter) in enumerate(zip(locations_1, locations_2, diameters)):
        if translation is not None:
            dx, dy, dz = translation
            x1, y1, z1 = x1 + dx, y1 + dy, z1 + dz
            x2, y2, z2 = x2 + dx, y2 + dy, z2 + dz

        line = vtk.vtkLineSource()
        line.SetPoint1(x1, y1, z1)
        line.SetPoint2(x2, y2, z2)
        line.SetResolution(0)
        line.Update()
        dv = (np.pi * (diameter/2) ** 2) * ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) ** 0.5
        volume += dv

        radius = diameter / 2
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(line.GetOutput())
        tube.SetRadius(radius)
        tube.SetNumberOfSides(resolution)

        data.AddInputConnection(tube.GetOutputPort())
    print(f"Volume {volume}, {len(diameters)} struts")

    if show_bounding_box:
        outline = vtk.vtkOutlineSource()
        outline.SetBounds(0, 10, 0, 10, 0, 10)
        data.AddInputConnection(outline.GetOutputPort())

    # # Add spheres at the 8 corners.
    # x, y, z = list(zip(*(locations_1 + locations_2)))
    # for x in [6, 10]: #range(np.max(x) + 1):
    #     for y in [6, 15]: #range(np.max(y) + 1):
    #         for z in [6, 10]: #range(np.max(z) + 1):
    #             d = 0
    #             for node_1, node_2 in zip(locations_1, locations_2):
    #                 if (x, y, z) == node_1 or (x, y, z) == node_2:
    #                     d = max(d, diameter)

    #             # Add a sphere with the same size as the largest strut connected to this node.
    #             sphere = vtk.vtkSphereSource()
    #             sphere.SetCenter(x, y, z)
    #             sphere.SetRadius(d / 2)
    #             sphere.SetThetaResolution(20)
    #             sphere.SetPhiResolution(20)

    #             data.AddInputConnection(sphere.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(data.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def export_stl(actor: vtk.vtkActor, filename: str) -> None:
    """Export an actor to an STL file."""
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(filename)
    writer.SetInputData(actor.GetMapper().GetInput())
    writer.Write()

def visualize_actors(*actors, gui: bool=False):
    """Show an interactive visualization window or a GUI of the given actor(s)."""

    if gui:
        application = QApplication(sys.argv)
        gui = InferenceWindow()
        ren = gui.ren
        iren = gui.iren
        window = gui.renwin

    else:
        ren = vtk.vtkRenderer()
        window = vtk.vtkRenderWindow()
        window.SetSize(600, 600)
        window.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.SetRenderWindow(window)

    # Add each actor.
    for actor in actors:
        ren.AddActor(actor)

    # Add the axes widget.
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(iren)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    if gui:
        ren.ResetCamera()
        window.Render()
        gui.show()
        sys.exit(application.exec_())
    
    else:
        ren.GetActiveCamera().SetParallelProjection(True)
        ren.ResetCamera()
        ren.GetActiveCamera().Azimuth(45)
        ren.GetActiveCamera().Elevation(45)
        iren.Initialize()
        window.Render()
        iren.Start()


if __name__ == "__main__":
    from scipy.io import loadmat
    data = loadmat('top.mat')
    density = data['x']
    write_pickle(density, 'topology_optimization.pickle')