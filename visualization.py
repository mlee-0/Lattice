import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # type: ignore

from datasets import *
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

    ren = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(window)

    array = np.array(array)
    
    points = vtk.vtkPoints()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("colors")
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            for z in range(array.shape[2]):
                if array[x, y, z] > 0:
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
    ren.GetActiveCamera().SetParallelProjection(True)
    ren.GetActiveCamera().Azimuth(45)
    ren.GetActiveCamera().Elevation(45)
    ren.ResetCamera()
    iren.Initialize()
    window.Render()
    iren.Start()

def visualize_nodes(array: np.ndarray, opacity: float=1.0) -> None:
    """Show an interactive visualization window of a 3D voxel model with white cubes representing nodes."""

    ren = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(window)

    data = vtk.vtkAppendPolyData()
    array = np.array(array)
    for x, y, z in np.argwhere(array):
        cube = vtk.vtkCubeSource()
        cube.SetBounds(x, x+1, y, y+1, z, z+1)
        cube.Update()
        data.AddInputData(cube.GetOutput())
    data.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor([1, 1, 1])
    actor.GetProperty().SetLineWidth(1)
    actor.GetProperty().SetOpacity(opacity)

    ren.AddActor(actor)
    ren.GetActiveCamera().SetParallelProjection(True)
    iren.Initialize()
    window.Render()
    iren.Start()

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

def visualize_lattice(locations_1: List[Tuple[float, float, float]], locations_2: List[Tuple[float, float, float]], diameters: List[float], true_diameters: List[float]=None) -> None:
    """Show an interactive visualization window of a lattice defined as a list of node 1 coordinates, a list of node 2 coordinates, and a list of diameters. All lists must be the same length."""

    assert len(locations_1) == len(locations_2) == len(diameters)

    ren = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(window)

    if true_diameters is not None:
        # The error value corresponding to red.
        max_error = 1 #np.abs(np.array(diameters) - np.array(true_diameters)).max()

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
        tube.SetNumberOfSides(3)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().SetLighting(False)
        
        # Set the color representing the magnitude of the error.
        if true_diameters is not None:
            error = abs(diameter - true_diameters[i])
            ratio = error / max_error

            color_error_0 = np.array([1, 1, 1])
            color_error_1 = np.array([1, 0, 0])
            color = ratio * color_error_1 + (1 - ratio) * color_error_0
            actor.GetProperty().SetColor(*color)

        ren.AddActor(actor)
    
    ren.GetActiveCamera().SetParallelProjection(True)
    ren.ResetCamera()
    iren.Initialize()
    window.Render()
    iren.Start()


if __name__ == "__main__":
    # lattice = convert_output_to_lattice(read_outputs(3)[2])
    # visualize_lattice(*lattice)

    # with open('Training_Data_10/outputs.pickle', 'rb') as f:
    #     outputs = pickle.load(f)
    # lattice = convert_adjacency_to_lattice(np.array(outputs[0, :, :]))
    # visualize_lattice(*lattice)
    
    # vector = read_pickle('Training_Data_10/outputs_vector.pickle')
    # lattice = convert_vector_to_lattice(vector[1000, :])
    # visualize_lattice(*lattice)

    inputs = read_pickle('Training_Data_10/inputs.pickle')
    visualize_input(inputs[0, 0, ...], opacity=1, length=0.9, use_lighting=True)

    # graphs = read_pickle('Training_Data_10/graphs.pickle')
    # lattice = convert_graph_to_lattice(graphs[0])
    # visualize_lattice(*lattice)
    
    # with open("Training_Data_10/inputs.pickle", 'rb') as f:
    #     array = pickle.load(f)
    # visualize_input(array[0, 0, ...], opacity=1.0)