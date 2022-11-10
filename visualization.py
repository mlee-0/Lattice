import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
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

def visualize_lattice(locations_1: List[Tuple[float, float, float]], locations_2: List[Tuple[float, float, float]], diameters: List[float], true_diameters: List[float]=None, screenshot_filename: str=None, gui: bool=False) -> None:
    """Show an interactive visualization window of a lattice defined as a list of node 1 coordinates, a list of node 2 coordinates, and a list of diameters. All lists must be the same length."""

    assert len(locations_1) == len(locations_2) == len(diameters)

    if gui:
        application = QApplication(sys.argv)
        gui = VisualizationWindow()
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
        tube.SetNumberOfSides(5)
        
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
    
    # Save the window as a PNG image. Must have a visualization running.
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


class VisualizationWindow(QMainWindow):
    """An GUI with an interactive visualizer and a sidebar with camera controls."""

    def __init__(self) -> None:
        super().__init__()

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(widget)

        layout.addWidget(self._sidebar())
        layout.addWidget(self._visualizer())

        # Start the interactor after the layout is created.
        self.iren.Initialize()
        self.iren.Start()
        self.ren.GetActiveCamera().Azimuth(45)
        self.ren.GetActiveCamera().Elevation(45)
    
    def _sidebar(self) -> QWidget:
        sidebar = QWidget()
        main_layout = QFormLayout(sidebar)
        main_layout.setAlignment(Qt.AlignTop)

        layout = QHBoxLayout()
        self.button_decrease_azimuth = QPushButton('−')
        self.button_decrease_azimuth.clicked.connect(self.azimuth)
        self.button_increase_azimuth = QPushButton('+')
        self.button_increase_azimuth.clicked.connect(self.azimuth)
        layout.addWidget(self.button_decrease_azimuth)
        layout.addWidget(self.button_increase_azimuth)
        main_layout.addRow('Azimuth', layout)
        
        layout = QHBoxLayout()
        self.button_decrease_elevation = QPushButton('−')
        self.button_decrease_elevation.clicked.connect(self.elevation)
        self.button_increase_elevation = QPushButton('+')
        self.button_increase_elevation.clicked.connect(self.elevation)
        layout.addWidget(self.button_decrease_elevation)
        layout.addWidget(self.button_increase_elevation)
        main_layout.addRow('Elevation', layout)

        return sidebar

    def _visualizer(self) -> QWidget:
        self.ren = vtk.vtkRenderer()
        widget = QVTKRenderWindowInteractor(self)
        self.renwin = widget.GetRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.iren = self.renwin.GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.ren.GetActiveCamera().SetParallelProjection(True)
        self.ren.GetActiveCamera().SetClippingRange(0.01, 1000)

        return widget
    
    def azimuth(self):
        camera = self.ren.GetActiveCamera()
        if self.sender() is self.button_decrease_azimuth:
            camera.Azimuth(-5)
        elif self.sender() is self.button_increase_azimuth:
            camera.Azimuth(+5)

        self.iren.Render()
    
    def elevation(self):
        camera = self.ren.GetActiveCamera()
        if self.sender() is self.button_decrease_elevation:
            camera.Elevation(-10)
        if self.sender() is self.button_increase_elevation:
            camera.Elevation(+10)

        self.iren.Render()


if __name__ == "__main__":
    # inputs = read_pickle('Training_Data_10/inputs.pickle')
    # visualize_input(inputs[0, 0, ...], opacity=1, length=1.0, use_lighting=not True)

    # lattice = convert_output_to_lattice(read_outputs(3)[2])
    # visualize_lattice(*lattice)

    # graphs = read_pickle('Training_Data_10/graphs.pickle')
    # lattice = convert_graph_to_lattice(graphs[0])
    # visualize_lattice(*lattice)

    # Test the difference in pixel values between PNG and JPG. Create an image with random noise and save as both PNG and JPG.
    import numpy as np
    from PIL import Image
    np.random.seed(45)
    a = np.random.rand(11, 11)
    a -= a.min()
    a /= a.max()
    a *= 255
    image = Image.fromarray(a.astype(np.uint8))
    image.save('test.png')
    image.save('test.jpg')
    with Image.open('test.png', 'r') as f:
        png = np.asarray(f, dtype=np.uint8).astype(float)
    with Image.open('test.jpg', 'r') as f:
        jpg = np.asarray(f, dtype=np.uint8).astype(float)
    
    d = np.abs(png - jpg) / 255
    print(f"Difference: {np.mean(d):.3f} (average), {np.min(d):.3f} (min), {np.max(d):.3f} (max)")