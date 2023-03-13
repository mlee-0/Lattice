"""
Functions for generating figures and visualizations.
"""


import matplotlib.pyplot as plt
import vtk

from preprocessing import DIRECTIONS, read_pickle
from visualization import *


def visualize_input():
    """Show a relative density."""
    return

def visualize_output():
    """Show a lattice structure."""

    data = read_pickle('Training_Data_11/outputs_array_augmented.pickle').numpy()
    data = data[0, ...]
    locations_1, locations_2, diameters = convert_array_to_lattice(data)
    visualize_lattice(
        locations_1=locations_1,
        locations_2=locations_2,
        diameters=diameters,
        resolution=10,
        gui=True,
    )   

def visualize_unit_cell():
    """Show the 26 struts extending from each node."""

    visualize_lattice(
        locations_1=[(0, 0, 0)]*26,
        locations_2=[(sign*x, sign*y, sign*z) for x, y, z in DIRECTIONS for sign in (-1, +1)],
        diameters=[0.25]*26,
        resolution=50,
        gui=True,
    )

def visualize_triangulation():
    """Show a triangulated surface that represents the design region."""

    coordinates = [
        [3.66753731481025,3.62644946029142,3.18828934024609],
        [6.45149740496001,6.66755825103741,9.91782470686778],
        [6.17067153172898,3.48291992155492,7.54406706214415],
        [3.37686690207595,1.23768953391515,3.25558115809498],
        [1.65547562278605,4.6642818819757,3.78191796177453],
        [9.44976283183825,0.400684963818208,5.68418743629346],
        [5.09960784755966,5.46721674392765,7.32355966885137],
        [9.53713444243099,9.62809058968873,4.09950908282565],
        [4.69904386311318,1.8459272251404,6.13652364049321],
        [9.32901147267347,0.420230229477961,9.72009007945684],
        [7.91717329677493,4.45642015571468,8.56158915809978],
        [4.94275965284496,3.9574147775316,3.84891746645172],
        [0.136719462301893,9.70176244719239,2.73125572174325],
        [2.40215066301545,6.69351094762089,8.34462237869957],
        [4.3788102232242,7.30173335721901,4.92245305798411],
        [8.87978273197626,1.09654346097627,4.70744846029491],
        [9.13219356930077,4.9496659304712,1.45237615021405],
        [6.23656185378252,9.25428005541289,1.00290112106909],
        [4.69212616416648,6.7213719895248,3.02960357257309],
        [8.24150540766324,4.23362401061368,9.83454320262463],
    ]
    triangulation = [
        [12,15,7],
        [4,9,5],
        [4,5,1],
        [3,7,9],
        [12,7,3],
        [1,12,4],
        [19,15,12],
        [14,5,7],
        [15,14,7],
        [15,5,14],
        [7,5,9],
        [1,19,12],
        [1,5,19],
        [13,5,15],
        [19,13,15],
        [19,5,13],
        [12,3,16],
        [2,7,3],
        [11,3,7],
        [11,7,2],
        [4,16,9],
        [12,16,4],
        [18,15,8],
        [19,8,15],
        [19,15,18],
        [18,8,17],
        [19,17,8],
        [19,18,17],
        [10,20,3],
        [10,11,20],
        [2,3,20],
        [11,2,20],
        [16,3,11],
        [16,11,6],
        [6,11,10],
        [10,3,9],
        [6,10,9],
        [16,6,9],
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

    visualize_actor_gui(actor)


if __name__ == '__main__':
    visualize_output()