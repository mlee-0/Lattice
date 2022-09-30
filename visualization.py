import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

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

def visualize_input(array: np.ndarray, opacity: float=0.5) -> None:
    """Start a visualization of a 3D input image."""

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
        for y in range(array.shape[0]):
            for z in range(array.shape[0]):
                if array[x, y, z] > 0:
                    points.InsertNextPoint(x, y, z)
                    colors.InsertNextTuple([array[x, y, z]] * 3)

    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.GetPointData().AddArray(colors)

    glyph = vtk.vtkCubeSource()
    # glyph.SetBounds([-0.4, 0.4] * 3)
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
    actor.GetProperty().SetLighting(False)

    ren.AddActor(actor)
    iren.Initialize()
    window.Render()
    iren.Start()

def visualize_nodes(array: np.ndarray, opacity: float=1.0) -> None:
    """Start a visualization a 3D voxel model with white cubes representing nodes."""

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
    iren.Initialize()
    window.Render()
    iren.Start()

def visualize_lattice_from_adjacency(lattice: np.ndarray) -> None:
    """Start a visualization of a lattice defined as a 2D array of diameters with shape (number of nodes, number of struts per node)."""

    ren = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(window)

    coordinates = read_coordinates()
    node_numbers = make_node_numbers()

    data = vtk.vtkAppendPolyData()

    for row in range(lattice.shape[0]):
        for column in range(lattice.shape[1]):
            d = lattice[row, column]

            if d > 0:
                x1, y1, z1 = np.unravel_index(row, shape=(11-1, 11-1, 11-1))
                x2, y2, z2 = np.unravel_index(column + 1, shape=(2, 2, 2)) + np.array([x1, y1, z1])
                node_1, node_2 = node_numbers[x1, y1, z1], node_numbers[x2, y2, z2]
                # strut = node_1 * lattice.shape[1] + node_2
                # node_1, node_2 = struts[strut]

                line = vtk.vtkLineSource()
                line.SetPoint1(coordinates[node_1 - 1])
                line.SetPoint2(coordinates[node_2 - 1])
                line.SetResolution(0)
                line.Update()
                data.AddInputData(line.GetOutput())
    data.Update()
    
    # tube = vtk.vtkTubeFilter()
    # tube.SetInputData(data.GetOutput())
    # tube.SetVaryRadiusToVaryRadiusByScalar(True)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLighting(False)

    ren.AddActor(actor)
    ren.ResetCamera()
    iren.Initialize()
    window.Render()
    iren.Start()

def visualize_lattice_from_graph(graph) -> None:
    """Start a visualization of a lattice defined as a bidirected graph."""

    ren = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(window)

    data = vtk.vtkAppendPolyData()

    # Skip duplicate edges.
    for i in range(graph.edge_index.size(1) // 2):
        node_1, node_2 = graph.edge_index[:, i]
        x1, y1, z1 = graph.x[node_1, 1:4]
        x2, y2, z2 = graph.x[node_2, 1:4]
        # if x1 == 0 and y1 == 0 and z1 == 0:
        #     print(node_1, x1, y1, z1)
        # if x2 == 0 and y2 == 0 and z2 == 0:
        #     print(node_2, x2, y2, z2)

        line = vtk.vtkLineSource()
        line.SetPoint1(x1, y1, z1)
        line.SetPoint2(x2, y2, z2)
        line.SetResolution(0)
        line.Update()
        data.AddInputData(line.GetOutput())
    data.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLighting(False)

    ren.AddActor(actor)
    ren.ResetCamera()
    iren.Initialize()
    window.Render()
    iren.Start()


if __name__ == "__main__":
    with open('Training_Data_10/outputs.pickle', 'rb') as f:
        outputs = pickle.load(f)
    visualize_lattice_from_adjacency(np.array(outputs[0, :, :]))

    with open("Training_Data_10/graphs.pickle", 'rb') as f:
        graphs = pickle.load(f)
    visualize_lattice_from_graph(graphs[0])
    
    # with open("Training_Data_10/inputs.pickle", 'rb') as f:
    #     array = pickle.load(f)
    # visualize_input(array[0, 0, ...], opacity=1.0)