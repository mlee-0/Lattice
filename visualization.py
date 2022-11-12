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

def visualize_input(array: np.ndarray, opacity: float=0.5, length: float=1.0, use_lighting: bool=False) -> None:
    """Show an interactive visualization window of a 3D input image with values in [0, 255]."""

    application = QApplication(sys.argv)
    gui = InferenceWindow()
    ren = gui.ren
    window = gui.renwin

    array = np.array(array)
    
    points = vtk.vtkPoints()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("colors")
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            for z in range(array.shape[2]):
                if True: #array[x, y, z] > 0:
                    points.InsertNextPoint(x, y, z)
                    colors.InsertNextTuple([array[x, y, z]] * 3)

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
    ren.AddActor(actor)

    ren.ResetCamera()
    window.Render()
    gui.show()
    sys.exit(application.exec_())

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

def convert_adjacency_to_lattice(array: np.ndarray) -> Tuple[list, list, list]:
    """Convert a 2D adjacency matrix into a tuple of coordinates and diameters."""

    coordinates = read_coordinates()
    node_numbers = make_node_numbers()

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

def make_actor_lattice(locations_1: List[Tuple[float, float, float]], locations_2: List[Tuple[float, float, float]], diameters: List[float], resolution: int=5):
    """Return an actor of a lattice."""

    data = vtk.vtkAppendPolyData()

    for i, ((x1, y1, z1), (x2, y2, z2), diameter) in enumerate(zip(locations_1, locations_2, diameters)):
        line = vtk.vtkLineSource()
        line.SetPoint1(x1, y1, z1)
        line.SetPoint2(x2, y2, z2)
        line.SetResolution(0)
        line.Update()

        radius = diameter / 2
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(line.GetOutput())
        tube.SetRadius(radius)
        tube.SetNumberOfSides(resolution)

        data.AddInputConnection(tube.GetOutputPort())
        
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(data.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def visualize_lattice(locations_1: List[Tuple[float, float, float]], locations_2: List[Tuple[float, float, float]], diameters: List[float], screenshot_filename: str=None, gui: bool=False) -> None:
    """Show an interactive visualization window of a lattice defined as a list of node 1 coordinates, a list of node 2 coordinates, and a list of diameters. All lists must be the same length."""

    assert len(locations_1) == len(locations_2) == len(diameters)

    if gui:
        application = QApplication(sys.argv)
        gui = InferenceWindow()
        ren = gui.ren
        window = gui.renwin
    else:
        ren = vtk.vtkRenderer()
        window = vtk.vtkRenderWindow()
        window.SetSize(600, 600)
        window.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.SetRenderWindow(window)

    data = vtk.vtkAppendPolyData()

    for i, ((x1, y1, z1), (x2, y2, z2), diameter) in enumerate(zip(locations_1, locations_2, diameters)):
        line = vtk.vtkLineSource()
        line.SetPoint1(x1, y1, z1)
        line.SetPoint2(x2, y2, z2)
        line.SetResolution(0)
        line.Update()

        radius = diameter / 2
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(line.GetOutput())
        tube.SetRadius(radius)
        tube.SetNumberOfSides(4)

        data.AddInputConnection(tube.GetOutputPort())
        
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(data.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetLighting(False)
    ren.AddActor(actor)
    
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
    
    # Save the window as a PNG image. Must have a visualization running first.
    if screenshot_filename is not None:
        filter = vtk.vtkWindowToImageFilter()
        filter.SetInput(window)
        # filter.SetScale(1)
        filter.SetInputBufferTypeToRGB()
        filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(os.path.join('Screenshots', str(screenshot_filename) + '.png'))
        writer.SetInputConnection(filter.GetOutputPort())
        writer.Write()


if __name__ == "__main__":
    inputs = read_pickle('Training_Data_10/inputs.pickle')
    visualize_input(inputs[0, 0, ...], opacity=1, length=1.0, use_lighting=not True)

    # lattice = convert_output_to_lattice(read_outputs(3)[2])
    # visualize_lattice(*lattice)

    # graphs = read_pickle('Training_Data_10/graphs.pickle')
    # lattice = convert_graph_to_lattice(graphs[0])
    # visualize_lattice(*lattice)