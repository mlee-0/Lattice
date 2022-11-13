"""
Run this script to run the main script as a GUI.
"""


import os
from queue import Queue
import sys
import threading
import time
from typing import Tuple

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QMargins
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import *
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # type: ignore

import main
from models import *
from datasets import *
from visualization import *


class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Queues used to communicate between threads.
        self.queue = Queue()
        self.queue_to_main = Queue()

        # Font objects.
        self.font_small = QFont()
        self.font_small.setPointSize(10)

        # Margin objects.
        self.margins_small = QMargins(5, 5, 5, 5)

        # Menu bar.
        menu_bar = self.menuBar()
        menu_view = menu_bar.addMenu("View")
        menu_help = menu_bar.addMenu("Help")
        self.action_toggle_console = menu_view.addAction("Show Console", self.toggle_console)
        self.action_toggle_console.setCheckable(True)
        self.action_toggle_console.setChecked(False)
        # self.action_toggle_status_bar = menu_view.addAction("Show Status Bar", self.toggle_status_bar)
        # self.action_toggle_status_bar.setCheckable(True)
        # self.action_toggle_status_bar.setChecked(True)
        # menu_view.addSeparator()
        # self.action_toggle_loss = menu_view.addAction("Show Current Loss Only")
        # self.action_toggle_loss.setCheckable(True)
        # menu_help.addAction("About...", self.show_about)

        # # Status bar.
        # self.status_bar = self.statusBar()
        # self.label_status = QLabel()
        # self.status_bar.addWidget(self.label_status)

        # # Automatically send console messages to the status bar.
        # # https://stackoverflow.com/questions/44432276/print-out-python-console-output-to-qtextedit
        # class Stream(QtCore.QObject):
        #     newText = QtCore.pyqtSignal(str)
        #     def write(self, text):
        #         self.newText.emit(str(text))
        # sys.stdout = Stream(newText=self.update_console)

        # Central widget.
        main_widget = QWidget()
        main_layout = QGridLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        self.sidebar = self._sidebar()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.sidebar)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Console.
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(self.font_small)
        self.console.setVisible(False)

        tabs = QTabWidget()
        tabs.addTab(self._training_tab(), "Training")
        tabs.addTab(self._testing_tab(), "Testing")
        
        layout_results = QVBoxLayout()
        layout_results.addWidget(self._progress_bar())
        layout_results.addWidget(tabs)

        # main_layout.addWidget(scroll_area, 0, 0)
        main_layout.addWidget(self.console, 1, 0)
        main_layout.addLayout(layout_results, 0, 1, 2, 1)
        main_layout.setRowStretch(0, 5)
        main_layout.setRowStretch(1, 0)
        main_layout.setColumnStretch(1, 1)
        
        # Timer that checkes the queue for information from main thread.
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.setInterval(100)
    
    def _sidebar(self) -> QWidget:
        """Return a widget containing fields for adjusting settings."""
        layout_sidebar = QVBoxLayout()
        layout_sidebar.setContentsMargins(0, 0, 0, 0)
        layout_sidebar.setAlignment(Qt.AlignTop)
        widget = QWidget()
        widget.setLayout(layout_sidebar)

        return widget

    def _progress_bar(self) -> QWidget:
        """Return a widget containing a progress bar."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self.button_start = QPushButton("Start")
        self.button_start.clicked.connect(self.on_start)
        layout.addWidget(self.button_start)

        self.label_progress = QLabel()
        self.label_progress_secondary = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.on_stop)
        self.button_stop.setEnabled(False)
        self.button_stop.setToolTip("Stop after current epoch.")
        
        layout.addWidget(self.label_progress)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.button_stop)

        return widget

    def _training_tab(self) -> QWidget:
        """Return a widget to be used as the training tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self.figure_loss = Figure()
        self.canvas_loss = FigureCanvasQTAgg(self.figure_loss)
        layout.addWidget(self.canvas_loss)

        self.figure_metrics = Figure()
        self.canvas_metrics = FigureCanvasQTAgg(self.figure_metrics)
        # layout.addWidget(self.canvas_metrics)

        return widget
    
    def _testing_tab(self) -> QWidget:
        """Return a widget to be used as the testing tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margins_small)
        layout.setAlignment(Qt.AlignTop)

        # Label that shows evaluation metrics.
        self.table_metrics = QTableWidget(0, 2)
        self.table_metrics.horizontalHeader().hide()
        self.table_metrics.verticalHeader().hide()
        self.table_metrics.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # Fill available horizontal space
        layout.addWidget(self.table_metrics)

        layout.addStretch(1)

        return widget
    
    def on_start(self):
        """Start training or testing."""
        self.button_start.setEnabled(False)
        self.console.clear()
        self.progress_bar.setRange(0, 0)
        self.button_stop.setEnabled(True)
        self.table_metrics.setRowCount(0)
        self.table_metrics.clear()

        main.kwargs["queue"] = self.queue
        main.kwargs["queue_to_main"] = self.queue_to_main
        main.kwargs["visualize_results"] = False

        self.thread = threading.Thread(
            target=main.main,
            kwargs=main.kwargs,
        )
        self.thread.start()
        self.timer.start()
    
    def on_stop(self):
        """Stop training after the current epoch has ended."""
        self.button_stop.setEnabled(False)
        self.queue_to_main.put(True)
    
    def update_console(self, text):
        """Display text in the console box."""
        if text.isprintable():
            self.console.insertPlainText(text)
            self.console.insertPlainText("\n")

    def toggle_status_bar(self):
        """Toggle visibility of status bar."""
        self.status_bar.setVisible(self.action_toggle_status_bar.isChecked())
    
    def toggle_console(self):
        """Toggle visibility of console."""
        self.console.setVisible(self.action_toggle_console.isChecked())

    def show_about(self):
        """Show a window displaying the README file."""
        with open("README.md", "r") as f:
            text = f.read()

        text_edit = QTextEdit(readOnly=True)
        text_edit.setMarkdown(text)

        window = QMainWindow(self)
        window.setWindowTitle("About")
        window.setCentralWidget(text_edit)
        window.show()

    def plot_loss(self, epochs, training_loss, previous_training_loss, validation_loss, previous_validation_loss):
        self.figure_loss.clear()
        axis = self.figure_loss.add_subplot(1, 1, 1)  # Number of rows, number of columns, index
        
        all_training_loss = [*previous_training_loss, *training_loss]
        all_validation_loss = [*previous_validation_loss, *validation_loss]
        if training_loss:
            axis.plot(range(1, len(all_training_loss)+1), all_training_loss, ".-", label="Training")
            axis.annotate(f"{training_loss[-1]:,.3e}", (epochs[len(training_loss)-1], training_loss[-1]), fontsize=10)
        if validation_loss:
            axis.plot(range(1, len(all_validation_loss)+1), all_validation_loss, ".-", label="Validation")
            axis.annotate(f"{validation_loss[-1]:,.3e}", (epochs[len(validation_loss)-1], validation_loss[-1]), fontsize=10)

        if previous_training_loss or previous_validation_loss:
            axis.vlines(epochs[0] - 0.5, 0, max(previous_training_loss + previous_validation_loss), label="Current session starts")
        
        axis.legend()
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Loss")
        axis.grid(axis="y")
        self.canvas_loss.draw()
    
    def plot_metrics(self):
        NUMBER_COLUMNS = 2
        self.figure_metrics.clear()
        # axis = self.figure_metrics.add_subplot(, NUMBER_COLUMNS, 1)

        self.canvas_metrics.draw()
    
    def check_queue(self):
        while not self.queue.empty():
            info = self.queue.get()
            progress_epoch: Tuple[int, int] = info.get("progress_epoch", (0, 0))
            progress_batch: Tuple[int, int] = info.get("progress_batch", (0, 0))
            epochs = info.get("epochs", range(0))
            training_loss = info.get("training_loss", [])
            previous_training_loss = info.get("previous_training_loss", [])
            validation_loss = info.get("validation_loss", [])
            previous_validation_loss = info.get("previous_validation_loss", [])
            info_metrics = info.get("info_metrics", {})

            # Update the progress label.
            strings_progress = []
            strings_progress.append(f"Epoch {progress_epoch[0]}/{progress_epoch[1]}")
            strings_progress.append(f"Batch {progress_batch[0]}/{progress_batch[1]}")
            text_progress = "\n".join(strings_progress)
            self.label_progress.setText(text_progress)

            # Update the progress bar.
            progress_value = max(progress_epoch[0]-1, 0) * progress_batch[1] + progress_batch[0]
            progress_max = max(progress_epoch[1], 1) * progress_batch[1]
            self.progress_bar.setValue(progress_value)
            self.progress_bar.setMaximum(progress_max)

            # Update the metrics.
            if info_metrics:
                for metric, value in info_metrics.items():
                    items = self.table_metrics.findItems(metric, Qt.MatchExactly)
                    if items:
                        row = self.table_metrics.row(items[0])
                    else:
                        row = self.table_metrics.rowCount()
                        self.table_metrics.insertRow(row)
                    self.table_metrics.setItem(row, 0, QTableWidgetItem(metric))
                    self.table_metrics.setItem(row, 1, QTableWidgetItem(str(value)))
                self.table_metrics.resizeRowsToContents()
                self.table_metrics.resizeColumnsToContents()

            if training_loss or previous_training_loss or validation_loss or previous_validation_loss:
                all_training_loss = [*previous_training_loss, *training_loss]
                all_validation_loss = [*previous_validation_loss, *validation_loss]
                all_epochs = [*range(1, epochs[0]), *epochs[:len(training_loss)]]
                self.plot_loss(
                    epochs = all_epochs,
                    training_loss = training_loss,
                    previous_training_loss = previous_training_loss,
                    validation_loss = validation_loss,
                    previous_validation_loss = previous_validation_loss,
                    # labels = ("Training", "Validation"),
                    # start_epoch = epochs[0] if previous_training_loss else None,
                )
                self.canvas_loss.draw()
                
        # Thread has stopped.
        if not self.thread.is_alive():
            self.button_start.setEnabled(True)
            self.button_stop.setEnabled(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.reset()
            self.timer.stop()
    
    # def closeEvent(self, event):
    #     """Base class method that closes the application."""
    #     # Return to default.
    #     sys.stdout = sys.__stdout__
    #     super().closeEvent(event)

class InferenceWindow(QMainWindow):
    """A GUI for testing a pretrained model on a custom dataset."""

    def __init__(self) -> None:
        super().__init__()

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(widget)

        layout.addWidget(self._sidebar(), stretch=0)
        layout.addWidget(self._visualizer(), stretch=1)

        # Start the interactor after the layout is created.
        self.iren.Initialize()
        self.iren.Start()
        self.reset()

        self.dataset = None
        self.generator = None
        self.actor = None
        self.batch = -1
    
    def _sidebar(self) -> QWidget:
        sidebar = QWidget()
        main_layout = QVBoxLayout(sidebar)
        main_layout.setAlignment(Qt.AlignTop)

        # Fields related to the density and lattice.
        box = QGroupBox('Setup')
        layout = QFormLayout(box)
        main_layout.addWidget(box)

        self.field_density_height = QSpinBox()
        self.field_density_width = QSpinBox()
        self.field_density_depth = QSpinBox()
        self.field_density_height.setRange(5, 100)
        self.field_density_width.setRange(5, 100)
        self.field_density_depth.setRange(5, 100)
        self.field_density_height.setValue(20)
        self.field_density_width.setValue(20)
        self.field_density_depth.setValue(20)
        layout_ = QHBoxLayout()
        layout_.setSpacing(0)
        layout_.addWidget(self.field_density_height)
        layout_.addWidget(self.field_density_width)
        layout_.addWidget(self.field_density_depth)
        layout.addRow('Density size', layout_)

        self.field_density_function = QComboBox()
        self.field_density_function.addItems(['linear', 'sin', 'cos', 'exp', 'random'])
        layout.addRow('Density function', self.field_density_function)

        self.field_lattice_height = QSpinBox()
        self.field_lattice_width = QSpinBox()
        self.field_lattice_depth = QSpinBox()
        self.field_lattice_height.setRange(1, 100)
        self.field_lattice_width.setRange(1, 100)
        self.field_lattice_depth.setRange(1, 100)
        self.field_lattice_height.setValue(5)
        self.field_lattice_width.setValue(5)
        self.field_lattice_depth.setValue(5)
        layout_ = QHBoxLayout()
        layout_.setSpacing(0)
        layout_.addWidget(self.field_lattice_height)
        layout_.addWidget(self.field_lattice_width)
        layout_.addWidget(self.field_lattice_depth)
        layout.addRow('Lattice size', layout_)

        self.field_lattice_type = QComboBox()
        self.field_lattice_type.addItems(['rectangle', 'circle'])
        layout.addRow('Lattice type', self.field_lattice_type)

        # Fields related to the generation process.
        box = QGroupBox('Generation')
        layout = QFormLayout(box)
        main_layout.addWidget(box)

        self.field_model = QLineEdit('.pth')
        layout.addRow('Model', self.field_model)

        self.field_batch_size = QSpinBox()
        self.field_batch_size.setRange(1, 1000)
        layout.addRow('Batch size', self.field_batch_size)

        self.checkbox_screenshot = QCheckBox('Screenshot each batch')
        layout.addRow(self.checkbox_screenshot)

        self.button_generate = QPushButton('Generate')
        self.button_generate.clicked.connect(self.generate)
        self.button_clear = QPushButton('Clear')
        self.button_clear.clicked.connect(self.clear)
        layout_ = QHBoxLayout()
        layout_.setSpacing(0)
        layout_.addWidget(self.button_generate)
        layout_.addWidget(self.button_clear)
        layout.addRow(layout_)

        self.label_runtime = QLabel()
        self.label_runtime.setAlignment(Qt.AlignCenter)
        layout.addRow(self.label_runtime)

        # Fields related to the camera.
        box = QGroupBox('Camera')
        layout = QFormLayout(box)
        main_layout.addWidget(box)

        layout_ = QHBoxLayout()
        layout_.setSpacing(0)
        self.button_decrease_azimuth = QPushButton('−')
        self.button_decrease_azimuth.clicked.connect(self.azimuth)
        self.button_increase_azimuth = QPushButton('+')
        self.button_increase_azimuth.clicked.connect(self.azimuth)
        layout_.addWidget(self.button_decrease_azimuth)
        layout_.addWidget(self.button_increase_azimuth)
        layout.addRow('Horizontal', layout_)
        
        layout_ = QHBoxLayout()
        layout_.setSpacing(0)
        self.button_decrease_elevation = QPushButton('−')
        self.button_decrease_elevation.clicked.connect(self.elevation)
        self.button_increase_elevation = QPushButton('+')
        self.button_increase_elevation.clicked.connect(self.elevation)
        layout_.addWidget(self.button_decrease_elevation)
        layout_.addWidget(self.button_increase_elevation)
        layout.addRow('Vertical', layout_)

        button_reset = QPushButton('Reset')
        button_reset.clicked.connect(self.reset)
        layout.addRow(button_reset)

        return sidebar
    
    def _visualizer(self) -> QWidget:
        self.ren = vtk.vtkRenderer()
        widget = QVTKRenderWindowInteractor(self)
        self.renwin = widget.GetRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.iren = self.renwin.GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.ren.GetActiveCamera().SetParallelProjection(True)

        return widget
    
    def save_screenshot(self):
        filter = vtk.vtkWindowToImageFilter()
        filter.SetInput(self.renwin)
        # filter.SetScale(1)
        filter.SetInputBufferTypeToRGB()
        filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(os.path.join('Screenshots', f'{self.batch:03}.png'))
        writer.SetInputConnection(filter.GetOutputPort())
        writer.Write()

    def generate(self) -> None:
        """Generate and show one batch of data."""

        # Load the dataset.
        if self.dataset is None:
            self.dataset = InferenceDataset(
                density_shape=(
                    self.field_density_height.value(),
                    self.field_density_width.value(),
                    self.field_density_depth.value(),
                ),
                density_function=self.field_density_function.currentText(),
                lattice_shape=(
                    self.field_lattice_height.value(),
                    self.field_lattice_width.value(),
                    self.field_lattice_depth.value(),
                ),
                lattice_type=self.field_lattice_type.currentText(),
            )

            self.generator = main.infer(
                model=ResNet(),
                filename_model=self.field_model.text(),
                dataset=self.dataset,
                batch_size=self.field_batch_size.value(),
            )

        # Generate the next batch.
        try:
            tic = time.time()
            locations_1, locations_2, diameters = next(self.generator)
            toc = time.time()
            self.label_runtime.setText(f"Generated in {toc - tic:.2f} seconds.")
        except StopIteration:
            self.button_generate.setEnabled(False)
        else:
            self.batch += 1
            actor = make_actor_lattice(locations_1, locations_2, diameters)
            self.set_actor(actor)

            if self.batch == 0:
                self.reset()
        
        if self.checkbox_screenshot.isChecked():
            self.save_screenshot()
        
        self.iren.Render()
    
    def clear(self):
        """Remove generated data."""
        self.ren.RemoveActor(self.actor)
        self.dataset = None
        self.generator = None
        self.actor = None
        self.batch = -1

        self.button_generate.setEnabled(True)
        self.label_runtime.clear()
    
        self.iren.Render()

    def set_actor(self, actor: vtk.vtkActor):
        if self.actor is not None:
            self.ren.RemoveActor(self.actor)
        self.actor = actor
        self.ren.AddActor(self.actor)

    def reset(self):
        """Reset the camera."""
        camera = self.ren.GetActiveCamera()
        camera.SetPosition(0, 0, 100)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        camera.SetClippingRange(0.01, 1000)
        camera.Azimuth(45)
        camera.Elevation(45)
        self.ren.ResetCamera()
        self.iren.Render()
    
    def azimuth(self):
        """Increment the horizontal camera angle."""
        camera = self.ren.GetActiveCamera()
        if self.sender() is self.button_decrease_azimuth:
            camera.Azimuth(-5)
        elif self.sender() is self.button_increase_azimuth:
            camera.Azimuth(+5)

        self.iren.Render()
    
    def elevation(self):
        """Increment the vertical camera angle."""
        camera = self.ren.GetActiveCamera()
        if self.sender() is self.button_decrease_elevation:
            camera.Elevation(-5)
        if self.sender() is self.button_increase_elevation:
            camera.Elevation(+5)

        self.iren.Render()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    window = InferenceWindow()
    window.setWindowTitle("Lattice Generation")
    window.show()
    sys.exit(application.exec_())