import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor 

from datasets import *


def plot_nodes(array: np.ndarray) -> None:
    """Show a 3D plot of node locations."""
    transparency = 1.0

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.voxels(
        filled=(array > 0),
        facecolors=(1, 1, 1, transparency),
        linewidth=0.25,
        edgecolors=(1, 1, 1),
    )
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    # ax.set(xlim=axis_limits, ylim=axis_limits, zlim=axis_limits)
    plt.show()

def visualize_input(array: np.ndarray) -> None:
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
    actor.GetProperty().SetOpacity(0.5)

    ren.AddActor(actor)
    iren.Initialize()
    window.Render()
    iren.Start()

def visualize_nodes(array: np.ndarray) -> None:
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
    actor.GetProperty().SetOpacity(1.0)

    ren.AddActor(actor)
    iren.Initialize()
    window.Render()
    iren.Start()

def visualize_lattice(lattice: np.ndarray) -> None:
    ren = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(window)

    data = vtk.vtkAppendPolyData()
    coordinates = read_coordinates()
    struts = read_struts()
    lattice = lattice[:100_000, :]

    for i in range(lattice.shape[0]):
        if i % 10000 == 0:
            print(f"{i}/{lattice.shape[0]}...", end='\r')
        
        strut, d = lattice[i, :]
        if d > 0:
            node_1, node_2 = struts[int(strut) - 1]
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
    with open('Lattice_1.txt', 'r') as f:
        # Dimensions of the adjacency matrix.
        h = (51 - (3-1)) ** 3  # Total number of nodes, reduced to remove duplicate nodes
        w = (3**3) - 1  # Total number of struts per node in a 3x3x3 neighborhood
        # Ignore the header line.
        _ = f.readline()
        # Read all lines except the lines at the bottom containing duplicate struts.
        data = []
        for line in range(1, h*w + 1):
            strut, d = f.readline().strip().split(',')
            strut, d = int(strut), float(d)
            if d > 0:
                data.append([strut, d])

    visualize_lattice(np.array(data))

    # with open("outputs.pickle", 'rb') as f:
    #     nodes = pickle.load(f)
    # visualize_nodes(nodes[0, 0, ...])
    
    # with open("inputs.pickle", 'rb') as f:
    #     array = pickle.load(f)
    # visualize_input(array[0, 0, :20, :20, :20])