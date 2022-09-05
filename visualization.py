# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor 

from datasets import *


def plot_lattice(data) -> None:
    """Plot the graph, given as a 2D array with shape (number of struts, 2), containing strut numbers in the first column and diameters in the second column."""

    struts = read_struts()

    data = data[:50, :]

    graph = nx.Graph()
    nodes = set()
    edges = []

    for i in range(data.shape[0]):
        strut, d = data[i, :]
        if d > 0:
            node_1, node_2 = struts[int(strut) - 1]
            nodes.add(node_1)
            nodes.add(node_2)
            edges.append((node_1, node_2, d))

    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)

    plt.figure()
    nx.draw(graph, nodelist=[], width=[graph[u][v]['weight'] for u, v in graph.edges()])
    plt.show()

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
    lattice = lattice[:500, :]

    for i in range(lattice.shape[0]):
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
    actor.GetProperty().SetColor([1, 1, 1])
    actor.GetProperty().SetLineWidth(1)

    ren.AddActor(actor)
    ren.ResetCamera()
    ren.SetBackground([0, 0, 0])
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