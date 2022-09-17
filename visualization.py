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
                if array[x, y, z]/255 >= 0:
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
    indices = np.nonzero(array)
    for x, y, z in zip(*indices):
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

def visualize_lattice(lattice: np.ndarray) -> None:
    """Start a visualization of a lattice defined as a 2D array of diameters with shape (number of nodes, number of struts per node)."""

    ren = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(window)

    coordinates = read_coordinates()
    struts = read_struts()

    data = vtk.vtkAppendPolyData()

    for node_1 in range(lattice.shape[0]):
        for node_2 in range(lattice.shape[1]):
            # if i % 10000 == 0:
            #     print(f"{i}/{lattice.shape[0]}...", end='\r')
            
            d = lattice[node_1, node_2]
            if d > 0:
                strut = node_1 * lattice.shape[1] + node_2
                node_1, node_2 = struts[strut]
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
    actor.GetProperty().SetLineWidth(1)

    ren.AddActor(actor)
    ren.ResetCamera()
    iren.Initialize()
    window.Render()
    iren.Start()


if __name__ == "__main__":
    with open('Training_Data_50/outputs.pickle', 'rb') as f:
        outputs = f.load(f)
    visualize_lattice(np.array(outputs[0, ...]))

    # with open("Training_Data_50/outputs_nodes.pickle", 'rb') as f:
    #     nodes = pickle.load(f)
    # visualize_nodes(nodes[1, 0, ...], opacity=1.0)
    
    # with open("Training_Data_50/inputs.pickle", 'rb') as f:
    #     array = pickle.load(f)
    # visualize_input(array[1, 0, ...], opacity=1.0)