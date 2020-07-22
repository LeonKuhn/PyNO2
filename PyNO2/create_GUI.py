"""
This file builds the signal and slot mechanisms of the PyQt GUI.
It also defines the TripleROI classes, which allow synched ROIs over multiple PyqtGraph ImageView instances.
"""

import importlib
import os
from functools import partial
from pathlib import Path

from constants import calibration_factor_func

import pyqtgraph
import pyqtgraph.exporters
from PyQt5 import uic
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import *

from Settings import SettingsEntry, CamSettingsNew
from my_functions import *


def file_dialog(path='', for_open=True, fmt='', is_folder=False):
    """
    Opens a file dialog to chose a file or directory.
    :param path: Start path, as str
    :param for_open: Whether the chosen path is for opening or saving, as bool
    :param fmt: Format of the files to be shown in the file explorer, as str
    :param is_folder: Whether the final path is to a file or to a folder, as bool
    :return: the chosen path, as str
    """

    # Instantiate a file dialog
    dialog = QFileDialog()

    # Do not use native dialog or custom directory icons
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontUseCustomDirectoryIcons
    dialog.setOptions(options)

    dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

    # Looking for files or folders?
    if is_folder:
        dialog.setFileMode(QFileDialog.DirectoryOnly)
    else:
        dialog.setFileMode(QFileDialog.AnyFile)

    # Is the dialog for opening or saving?
    dialog.setAcceptMode(QFileDialog.AcceptOpen) if for_open else dialog.setAcceptMode(QFileDialog.AcceptSave)

    # Filter by specified file format
    if fmt != '' and is_folder is False:
        dialog.setDefaultSuffix(fmt)
        dialog.setNameFilters([f'{fmt} (*.{fmt})'])

    # Set the starting directory
    if path != '':
        dialog.setDirectory((str(Path(__file__).parents[0]) + "\\" + str(path)).replace("\\", "/"))
    else:
        pass

    if dialog.exec_() == QDialog.Accepted:
        path = dialog.selectedFiles()[0]  # returns a list
        return path
    else:
        return ''


"TripleROI classes; Allow synched ROIs over multiple PyqtGraph ImageView instances."


def ROI_positions(roi):
    """
    Returns the ROI coordinates as a dict
    :param roi: Roi to get coordinates from
    :return: dict
    """

    width, height = roi.size()
    position = tuple(roi.pos())

    bottom_left = position[0], position[1] + height
    top_right = bottom_left[0] + width, bottom_left[1] - height
    bottom_right = bottom_left[0] + width, bottom_left[1]
    top_left = bottom_left[0], bottom_left[1] - height

    return {
        "top-left": top_left,
        "top-right": top_right,
        "bottom-right": bottom_right,
        "bottom-left": bottom_left
    }


class TripleROI(QObject):
    """
    Class of ROIs that can be synched across multiple pyqtgraph.ImageView instances.
    """

    # Define signal, which will emit the image slices selected by the ROI as a list of np.ndarrays

    sig = pyqtSignal(list)

    def __init__(self, name, parents, color=1, roi_type=pyqtgraph.RectROI):
        """
        :param name: Display name of the ROI
        :param parents: Array of pyqtgraph.ImageView instances that share this ROI
        :param color: Color of this ROI
        """

        QObject.__init__(self)

        self.parents = parents
        self.pen = (color, 9)
        self.roi_type = roi_type
        self.name = name
        self.rois = []
        self.labels = []
        self.surpressed = False

        # Set the ROIs up:
        for i in range(len(parents)):

            # Create and add a ROI
            if self.roi_type != pyqtgraph.LineSegmentROI:
                self.rois.append(self.roi_type([100, color * 100], [100, 50], pen=self.pen, invertible=True))

                # Create and add a label
                self.labels.append(pyqtgraph.TextItem(html="<b>{}</b>".format(self.name)))
                self.labels[i].setColor(color)
                self.labels[i].setPos(self.rois[i].pos())

            else:
                self.rois.append(self.roi_type([[100, 50 * (color+ 3)], [350, 50 * (color + 3)]], pen=self.pen, invertible=True))

                # Create and add a label
                self.labels.append(pyqtgraph.TextItem(html="<b>{}</b>".format(self.name)))
                self.labels[i].setColor(color)
                self.labels[i].setPos(225, 50 * (color+ 3))


            # Connect the region change signal of each ROI with the synch function
            self.rois[i].sigRegionChangeFinished.connect(partial(self.synch, self.rois[i]))
            self.rois[i].sigRegionChangeFinished.connect(self.emit)

            # Finally, add the ROIs and labels to the pyqtgraph.ImageView instances
            parents[i].getView().addItem(self.rois[i])
            parents[i].getView().addItem(self.labels[i])

    def __call__(self, i):
        """
        :param i: Number of the ImageView instance to obtain image slice from (starts at 0).
        :return: np.ndarray
        """

        roi = self.rois[i]
        gv = self.parents[i]
        try:
            return roi.getArrayRegion(gv.getImageItem().image, gv.getImageItem())
        except:
            pass

    def synch(self, caller):
        """
        Synchs the ROIs across the different pyqtgraph.ImageView instances
        :param caller: ROI to which the other ROIs synch
        :return: None
        """

        for i in range(len(self.rois)):
            roi = self.rois[i]
            label = self.labels[i]

            # Set ROI position
            roi.setPos(caller.pos(), finish=False)
            roi.setSize(caller.size(), finish=False)
            roi.stateChanged(finish=False)

            # Set label position
            label.setPos(caller.pos())

        # Emit the new image slices
        self.emit()

    def emit(self):
        """
        Emits all three ROIs at once
        :return: None
        """

        # Exit if signal emission is surpessed:
        if self.surpressed:
            return

        self.sig.emit([self.__call__(0),
                      self.__call__(1),
                      self.__call__(2)])


class TripleLineROI(TripleROI):
    """
    Same as TripleROI class, but with a pyqtgraph.LineSegmentROI as ROI.
    """
    def __init__(self, name, parents, color=1):
        """
        :param name: Display name of the ROI
        :param parents: Array of pyqtgraph.ImageView instances that share this ROI
        :param color: Color of this ROI
        """

        super().__init__(name, parents, color, roi_type=pyqtgraph.LineSegmentROI)

    def synch(self, caller):
        """
        Synchs the ROIs across the different pyqtgraph.ImageView instances
        :param caller: ROI to which the other ROIs synch
        :return: None
        """

        """
        ROI synchronisation is a bit more complicated for the pyqtgraph.LineSegmentROIs. Therefore we delete
        and reinitialise all ROIs on every synch call, because it allows to set ROI positions by passing
        handle coordinates.
        """

        # First determine the handle coordinates
        x1 = int(caller.getLocalHandlePositions()[0][1].x())
        y1 = int(caller.getLocalHandlePositions()[0][1].y())
        x2 = int(caller.getLocalHandlePositions()[1][1].x())
        y2 = int(caller.getLocalHandlePositions()[1][1].y())

        for i in range(len(self.rois)):
            gv = self.parents[i]

            # Delete the ROI from the ImageView
            gv.getView().removeItem(self.rois[i])

            # Create a new ROI at the new position
            self.rois[i] = self.roi_type(positions=((x1, y1), (x2, y2)),
                                         pos=caller.pos(),
                                         pen=self.pen,
                                         invertible=True)

            # Add the new ROI
            gv.getView().addItem(self.rois[i])

            # Adjust label position
            self.labels[i].setPos(caller.pos() + (0.5 * (x1 + x2), 0.5 * (y1 + y2)))

            # Because the ROI has been deleted, the signal connection must be reestablished
            self.rois[i].sigRegionChangeFinished.connect(partial(self.synch, self.rois[i]))

        # Emit the new image slices
        self.emit()

    def get_plot_data(self, i):

        gv = self.parents[i]

        coordinates = self.rois[i].getArraySlice(gv.getImageItem().image, gv.getImageItem(), returnSlice=True)[0]

        x1 = int(coordinates[0].start)
        y1 = int(coordinates[1].start)
        x2 = int(coordinates[0].stop)
        y2 = int(coordinates[1].stop)

        arr = self.__call__(i)

        X, Y, S = np.linspace(x1, x2, len(arr)), np.linspace(y1, y2, len(arr)), arr

        return X, Y, S


class TripleCropROI(TripleROI):
    """
    Same as TripleROI class, but with extra information for cropping an image.
    """

    def __init__(self, name, parents, color=1):
        """
        :param name: Display name of the ROI
        :param parents: Array of pyqtgraph.ImageView instances that share this ROI
        :param color: Color of this ROI
        """

        super().__init__(name, parents, color)

        # Create additional labels. These help to specify the pixel offset and ROI dimensions
        self.extra_labels = []

        # Creat a sub-array for each parent
        for i in range(len(parents)):
            self.extra_labels.append([])

        # Iterate over the sub-arrays
        for i in range(len(self.extra_labels)):

            # Get corner positions as dict
            positions = ROI_positions(self.rois[i])

            # Now iterate over each label of that sub-array
            j = 0
            for key in positions:

                self.extra_labels[i].append(pyqtgraph.TextItem(html="<b>{}</b>".format(str(positions[key]))))
                self.extra_labels[i][j].setColor(color)
                self.extra_labels[i][j].setPos(*positions[key])

                parents[i].getView().addItem(self.extra_labels[i][j])

                j += 1

    def synch(self, caller):

        super().synch(caller)

        # The extra labels need to move as well
        for i in range(len(self.extra_labels)):

            # Get corner positions as dict
            positions = ROI_positions(caller)

            # Now iterate over each label of that sub-array
            j = 0
            for key in positions:

                label = self.extra_labels[i][j]

                label.setPos(*positions[key])
                label.setHtml("<b>{}</b>".format(str([int(x) for x in positions[key]])))
                j += 1

    def offset(self):
        """
        :return: The offset of this ROI, as intended by the x/y offset nodes of the camera's image format settings.
        """

        top_left = ROI_positions(self.rois[0])["top-left"]

        # Rules from SpinView: (offset_x) % 2 = 0 and (offset_y) % 2 = 0. So we adjust the values slightly:

        offset_x = int(top_left[0])
        offset_y = int(top_left[1])

        if offset_x % 2 != 0: offset_x += 1
        if offset_y % 2 != 0: offset_y += 1

        print("Offset ~~~~~~~~~~~~~", offset_x, offset_y)
        return offset_x, offset_y

    def dimensions(self):
        """
        :return: The dimensions of this ROI, as intended by the width/height nodes of the camera's image format settings.
        """

        width, height = self.rois[0].size()

        # Rules from SpinView: (width-4) % 4 = 0 and (height-2) % 2 = 0. So we adjust the values slightly:

        rest_width = (width-4) % 4
        width -= rest_width

        rest_height = (height - 2) % 2
        height -= rest_height

        return int(width), int(height)

    def emit(self):
        """
        Emits offset and dimensions of this ROI.
        """

        # Exit if signal emission is surpessed:
        if self.surpressed:
            return

        self.sig.emit([*self.offset(), *self.dimensions()])

"""
MainWindow class; Here, the Signal-Slot mechanisms are defined
"""
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, s):

        super().__init__()

        # Make the Settings instance available, so user input in the MainWindow can be saved internally
        self.s = s

        # Load UI from .ui file that has been generated with QT Designer
        uic.loadUi("GUI/gui2.ui", self)

        # Hide menu and ROI buttons of the graphicsView widgets
        self.graphicsView.ui.roiBtn.hide()
        self.graphicsView.ui.menuBtn.hide()

        self.graphicsView_2.ui.roiBtn.hide()
        self.graphicsView_2.ui.menuBtn.hide()

        self.graphicsView_3.ui.roiBtn.hide()
        self.graphicsView_3.ui.menuBtn.hide()

        """
        Add a new menu item to the ViewBox so images can be saved as .npy files. This can be used to generate
        background images.
        """
        # Add a new action to the ViewBox context menu
        save_as_npy_1 = self.graphicsView.getView().menu.addAction('Save as .npy file ...')
        save_as_npy_2 = self.graphicsView_2.getView().menu.addAction('Save as .npy file ...')
        save_as_npy_3 = self.graphicsView_3.getView().menu.addAction('Save as .npy file ...')

        # Define callback for action
        def save_as_npy_callback():

            save_path = file_dialog()

            for i, gv in enumerate([self.graphicsView, self.graphicsView_2, self.graphicsView_3]):
                # Retrieve image as .npy file:
                image_as_npy = gv.getImageItem().image

                # Open save path file dialogue and determine save path:
                save_path_sub = save_path + "_%s.npy" % i

                # Save the image
                np.save(save_path_sub, image_as_npy)

        # Connect actions to callbacks
        save_as_npy_1.triggered.connect(save_as_npy_callback)
        save_as_npy_2.triggered.connect(save_as_npy_callback)
        save_as_npy_3.triggered.connect(save_as_npy_callback)

        """
        Add a shortcut for setting a background image
        """
        # Add a new action to the ViewBox context menu
        set_as_background = self.graphicsView_3.getView().menu.addAction('Set as background image ...')

        # Define callback for action
        def set_as_background_callback():
            self.s.background_image.__set__(self.graphicsView_3.getImageItem().image)

        # Connect action to callback
        set_as_background.triggered.connect(set_as_background_callback)

        """
        Add a function for stack evaluation
        """

        # Load all scripts defined in the "evaluation_script/" directory
        eval_scripts = os.listdir("evaluation_scripts/")
        eval_scripts = [e[:-3] for e in eval_scripts]  # remove .py suffix for import statement
        eval_scripts.remove("__init__")                # remove __init__ because it is not a true evaluation script
        menu_entries = []

        # Add menu entries in the GUI
        for script in eval_scripts:
            menu_entries.append(self.graphicsView_3.getView().menu.addAction('Evaluate: {}'.format(script)))

        # Each of the new menu entries must be connected to its respective evaluation script
        def mass_export_callback(scr):

            """Evaluation scripts are loaded anew everytime they are called from the MainWindow - So it must be checked whether they are already
            loaded or not. If yes, they must be re-loaded, else they must be loaded."""

            global eval_script

            reload = True

            try:
                eval_script
            except NameError:
                reload = False

            if reload:
                try:
                    importlib.reload(eval_script)
                except Exception as e:
                    print("Error reloading module:", e)

            eval_script = importlib.import_module("evaluation_scripts.{}".format(scr))

            # User inputs the minimum and maximum index for the stack to be evaluated
            self.export_min_index = int(input("Minimum index: "))
            self.export_max_index = int(input("Maximum index: "))

            # Call the "prepare" routine of the evaluation script before iterating stack
            eval_script.prepare(self)

            # Set slider to 0, to start at the first image.
            self.horizontalSlider_stack_upper.setValue(max(0, self.export_min_index))

            while self.horizontalSlider_stack_upper.value() < min(self.export_max_index, len(self.s.sc1.image_names_list.v)-1):

                print("Calling eval script on {}/{}".format(self.horizontalSlider_stack_upper.value(), self.export_max_index))

                # Now, extract the evaluation routine for a single image from an evaluation script.
                eval_script.run(self)

                # Increase slider by 1
                self.horizontalSlider_stack_upper.setValue(self.horizontalSlider_stack_upper.value() + 1 + self.s.tbin.v)

            # Finally, call the "trail" routine of the evaluation script after the stack has been iterated over.
            eval_script.trail(self)

        # Connect actions with callbacks
        for i, entry in enumerate(menu_entries):

            entry.triggered.connect(partial(mass_export_callback, eval_scripts[i]))

        """
        Add a shortcut for exporting the plot window to a .csv file
        """
        # Add menu entries in the GUI
        export_plot = self.plot_item.getViewBox().menu.addAction('Export plot as .csv file ...')

        # Define callback for action
        def export_plot_callback():

            plot_item = self.plot_item.getPlotItem()
            plot_item.curves = [plot_item.curves[0]]
            plot_item.dataItems = [plot_item.dataItems[0]]

            exporter = pyqtgraph.exporters.CSVExporter(self.plot_item.getPlotItem())
            save_name = str(input("Enter a save name for this plot data: "))

            exporter.parameters()["separator"] = "tab"
            exporter.export("Plot_Exports/" + save_name + ".csv")

        # Connect actions with callbacks
        export_plot.triggered.connect(export_plot_callback)

        """
        Add a shortcut for plotting the contents of a directoriy with .csv files
        """
        # Add menu entries in the GUI
        plot_directory = self.plot_item.getViewBox().menu.addAction('Plot a directory of .csv files ...')

        # Define callback for action
        def plot_directory_callback():

            # Some plot formatting:
            def cm2inch(*tupl):
                inch = 2.54
                if isinstance(tupl[0], tuple):
                    return tuple(i / inch for i in tupl[0])
                else:
                    return tuple(i / inch for i in tupl)

            plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['font.family'] = 'STIXGeneral'

            plt.rcParams.update({'font.size': 10})
            plt.rcParams["figure.figsize"] = cm2inch(12, 12 / (1.618 * 1.5))

            # Load the csv files and plot them into a mutual figure
            directory = file_dialog("Plot_Exports/", for_open=True, is_folder=True)

            files = os.listdir(directory)

            for f in files:
                # If a plot_this.py file exists in the dictionary, use it to plot the data
                if f == "plot_this.py":
                    exec(open(directory + "/" + f).read())
                    break

                else:
                # Else, follow a standard plotting routine
                    try:
                        # Extract data
                        data = np.genfromtxt(directory + "/" + f, deletechars=",", skip_header=1)
                        ordinate = data[:, 1]
                        abscissa = data[:, 0]

                        # Extract plot parameters
                        params = f.split("_")
                        label, color, alpha, linestyle, _ = params

                        # Plot the data
                        plt.plot(abscissa, ordinate,
                                 color=color,
                                 label=label,
                                 alpha=float(alpha),
                                 linestyle=linestyle)

                    except Exception as e:
                        print("Could not plot data from file", directory + "/" + f)
                        print("Error:", e)

            plt.legend()

            plt.show()

        # Connect actions with callbacks
        plot_directory.triggered.connect(plot_directory_callback)

        # Define the array of pyqtgraph.ImageView instances that share the ROIs
        parents = [self.graphicsView, self.graphicsView_2, self.graphicsView_3]

        # Add the synchronised TripleROIs
        self.avg_ROI = TripleROI("Average and noise", parents, 1)
        self.parallax_ROI = TripleROI("Parallax correction", parents, 2)
        self.offset_ROI = TripleROI("Offset", parents, 3)
        self.Sc_ROI = TripleROI("Sc", parents, 4)
        self.AE_ROI = TripleROI("AE", parents, 5)
        self.plot_ROI = TripleLineROI("Plot ROI", parents, 9)

        # Set up the plot window for column and row plots
        # Plot window for the pure ROI plot
        self.plot_curve_item = pyqtgraph.PlotCurveItem()
        self.plot_curve_item.setPen((255, 255, 255, 130), width=1)
        self.plot_item.addItem(self.plot_curve_item)

        # Plot window for the "trend" plot (rolling mean along the pure ROI data)
        self.plot_curve_item_trend = pyqtgraph.PlotCurveItem()
        self.plot_curve_item_trend.setPen((255, 0, 0), width=2)
        self.plot_item.addItem(self.plot_curve_item_trend)

        # Enable grid and auto range
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.enableAutoRange()

        """Definition of the signal and slots mechanisms:"""
        # SLOT: Toggles the signal emission of the average/noise ROI, saves time if disabled
        def toggle_avg_noise_calculation(boolean):

            if boolean:
                # If the signals are supposed to be ignored
                self.avg_ROI.sig.disconnect()

                self.s.average.__set__(0)
                self.s.detection_limit.__set__(0)
                self.s.sc1.average.__set__(0)
                self.s.sc1.relative_noise.__set__(0)
                self.s.sc2.average.__set__(0)
                self.s.sc2.relative_noise.__set__(0)

            else:
                # If the signals are supposed to be sent
                self.avg_ROI.sig.connect(lambda x: self.s.sc1.average.__set__(np.nanmean(x[0])))
                self.avg_ROI.sig.connect(lambda x: self.s.sc2.average.__set__(np.nanmean(x[1])))
                self.avg_ROI.sig.connect(lambda x: self.s.average.__set__(np.nanmean(x[2])))
                self.avg_ROI.sig.connect(lambda x: self.s.sc1.relative_noise.__set__(np.nanstd(x[0]) / np.nanmean(x[0])))
                self.avg_ROI.sig.connect(lambda x: self.s.sc2.relative_noise.__set__(np.nanstd(x[1]) / np.nanmean(x[1])))
                self.avg_ROI.sig.connect(lambda x: self.s.detection_limit.__set__(np.nanstd(x[2])))

                def set_sc_and_calib(x):

                    # Determine S_c
                    if self.radio_scell_auto.isChecked():
                        print()
                        self.s.cell_cd.set_now(-np.log(np.nanstd(x[1]) / np.nanstd(x[0])) / mean_absorption)
                    else:
                        self.s.cell_cd.set_now(float(self.lineEdit_scell.text()))

                    # Determine calibration factor:
                    if self.radioButton_units_1.isChecked():
                        self.s.calibration_factor.set_now(1)

                    else:
                        self.s.calibration_factor.set_now(calibration_factor_func(self.s.cell_cd.v))

                self.Sc_ROI.sig.connect(set_sc_and_calib)
                self.Sc_ROI.emit()
                self.avg_ROI.emit()



        # CONNECTION: Ignore signals from ROIs if desired
        self.checkBox_ignore_ROI_signals.toggled.connect(lambda boolean: toggle_avg_noise_calculation(boolean))

        # CONNECTION: Let offset ROI write to offset field of settings instance:
        self.offset_ROI.sig.connect(lambda x: self.s.sc1.offset.__set__(np.mean(x[0])))
        self.offset_ROI.sig.connect(lambda x: self.s.sc2.offset.__set__(np.mean(x[1])))

        # SLOT: Plot along the "Plot ROI" line marker
        def plot_routine(arr):
            """
            Plots data retrieved by the "Plot ROI" marker to the plot window
            :param arr: data to plot
            :return:
            """

            coordinates = self.plot_ROI.rois[2].getArraySlice(self.graphicsView_3.getImageItem().image, self.graphicsView_3.getImageItem(), returnSlice=True)[0]

            x1 = int(coordinates[0].start)
            y1 = int(coordinates[1].start)
            x2 = int(coordinates[0].stop)
            y2 = int(coordinates[1].stop)

            axis_text = "Signal"
            if self.s.average.units != "": axis_text += "[{}]".format(self.s.average.units)

            self.plot_item.showGrid(self.s.grid.v, self.s.grid.v)

            if abs(x2 - x1) > abs(y2 - y1):
                # draw against x-axis:
                self.plot_item.setLabel("left", text=axis_text)
                self.plot_item.setLabel("bottom", text="Path projection to x-axis")

                if s.signal_axis_scale.v == "linear":
                    self.plot_item.setLogMode(x=False, y=False)

                elif s.signal_axis_scale.v == "logarithmic":
                    self.plot_item.setLogMode(x=False, y=True)

                self.plot_curve_item.setData(np.linspace(x1, x2, len(arr)), arr)

                trend = np.convolve(arr, np.ones((self.s.rolling_mean_width.v,)) / self.s.rolling_mean_width.v, mode="valid")

                self.plot_curve_item_trend.setData(np.linspace(x1, x2, len(trend)), trend)

            else:
                # draw against y-axis:
                self.plot_item.setLabel("left", text="Path projection to y-axis")
                self.plot_item.setLabel("bottom", text=axis_text)

                if s.signal_axis_scale.v == "linear":
                    self.plot_item.setLogMode(x=False, y=False)

                elif s.signal_axis_scale.v == "logarithmic":
                    self.plot_item.setLogMode(x=True, y=False)

                self.plot_curve_item.setData(np.flip(arr), np.linspace(y1, y2, len(arr)))

                trend = np.convolve(np.flip(arr), np.ones((self.s.rolling_mean_width.v,)) / self.s.rolling_mean_width.v, mode="valid")

                self.plot_curve_item_trend.setData(trend, np.linspace(y1, y2, len(trend)))

        # CONNECTION: Let the Plot ROI draw to the plot window
        self.plot_ROI.sig.connect(lambda x: plot_routine(x[self.s.plot_roi_input.v]))
        
        # CONNECTION: Control elements for background correction:
        self.radio_background_ROI.clicked.connect(lambda x: self.s.offset_source.__set__("ROI"))
        self.radio_background_manual.clicked.connect(lambda x: self.s.offset_source.__set__("Manual"))

        # CONNECTION: Toggle auto-limits for histogram of result image view:
        self.checkBox_auto_histogram.clicked.connect(lambda x: self.s.auto_histogram.__set__(x))

        # CONNECTION: Toggle showing processed measurement images:
        self.checkBox_show_processed.toggled.connect(lambda x: self.s.show_processed_images.__set__ (x))

        # CONNECTION: Toggle dark image correction:
        self.checkBox_dark_correction.clicked.connect(lambda x: self.s.dark_frame_correction.__set__(x))

        # CONNECTION: Toggle NLC:
        self.checkBox_NLC.clicked.connect(lambda x: self.s.NLC.__set__(x))

        # CONNECTION: Toggle flat field correction:
        self.checkBox_ffc.clicked.connect(lambda x: self.s.flat_field_correction.__set__(x))

        # Add all files in the "FFC/" directory to the FFC selection menu
        self.listWidget_ffc1.addItems([x for x in os.listdir("FFC/") if x[-4:] == ".npy"])
        self.listWidget_ffc2.addItems([x for x in os.listdir("FFC/") if x[-4:] == ".npy"])

        # CONNECTION: Save FFC selection in Settings instance
        self.listWidget_ffc1.itemClicked.connect(lambda x: self.s.sc1.ffc.__set__(FFC("FFC/" + x.text())))
        self.listWidget_ffc2.itemClicked.connect(lambda x: self.s.sc2.ffc.__set__(FFC("FFC/" + x.text())))

        # CONNECTION: Toggle parallax correction
        self.checkBox_parallax_correction.clicked.connect(lambda x: self.s.parallax_correction.__set__(x))

        # CONNECTION: Toggle offset correction:
        self.checkBox_offset_correction.clicked.connect(lambda x: self.s.offset_correction.__set__(x))

        # CONNECTION: Toggle background correction & dialogue:
        self.checkBox_background_correction.clicked.connect(lambda x: self.s.background_correction.__set__(x))

        # SLOT: Background image by user selection
        def background_image_dialogue():
            """
            Lets the user pick a background image via GUI
            :return: None
            """

            path = file_dialog("Live_View/", is_folder=False, fmt="npy")
            self.s.background_image.__set__(np.load(path))

        # CONNECTION: Background selection
        self.pushButton_pick_background_image.clicked.connect(background_image_dialogue)

        # CONNECTION: Background subtraction
        self.checkBox_background_correction.toggled.connect(lambda x: self.s.background_correction.__set__(x))

        # SLOT: User-defined mask to be subtracted from quotient image, written to the mask field of Settings instance
        def generate_mask(func_string):
            """
            This function extracts the user input from the mask function field and turns it into a function f(x,y) which
            defines a signal field to subtract from the quotient image.
            :param func_string: function return line as String
            :return: np.ndarray
            """

            mask = eval(func_string)
            self.s.mask.__set__(mask)

        # CONNECTIONS: Control elements for the plot window:
        self.checkBox_grid.toggled.connect(lambda x: self.s.grid.__set__(x))
        self.checkBox_plot_ROI_source_0.clicked.connect(lambda: self.s.plot_roi_input.__set__(0))
        self.checkBox_plot_ROI_source_1.clicked.connect(lambda: self.s.plot_roi_input.__set__(1))
        self.checkBox_plot_ROI_source_2.clicked.connect(lambda: self.s.plot_roi_input.__set__(2))
        self.spinBox_rolling_mean.valueChanged.connect(lambda x: self.s.rolling_mean_width.__set__(x))

        """Unit and calibration factor control:"""
        # SLOT: Update units of detecion_limit, average and offset field
        def set_units(units):
            # Sets the units for some fields in the settings instance when the user switches between instrument signal and column densities
            s.detection_limit.units = units
            s.average.units = units
            s.offset.units = units

        # CONNECTIONS: Controls for displayed Units
        self.radioButton_units_1.clicked.connect(lambda x: self.s.calibration_factor.__set__(1))
        self.radioButton_units_3.clicked.connect(lambda x: self.s.calibration_factor.__set__(calibration_factor_func(self.s.cell_cd.v)))
        self.radioButton_units_1.clicked.connect(partial(set_units, ""))
        self.radioButton_units_3.clicked.connect(partial(set_units, "molec cm^-2"))

        # CONNECTIONS: Controls for digital spatial binning
        self.spinBox_sbin_x.valueChanged.connect(lambda x: self.s.sbin_x.__set__(max([y for y in range(1, x + 1) if 1920 % y == 0])))
        self.spinBox_sbin_y.valueChanged.connect(lambda x: self.s.sbin_y.__set__(max([y for y in range(1, x + 1) if 1200 % y == 0])))

        # CONNECTIONS: Control for temporal binning:
        self.spinBox_tbin.valueChanged.connect(lambda x: self.s.tbin.__set__(x))
        self.spinBox_exposure_loops.valueChanged.connect(lambda x: self.s.exposure_loops.__set__(x))

        # SLOT: Enable On-Chip binning on the camera
        def radioButton_onchip_clicked(s, set_to):

            # Remember old value (for t_exp correction):
            old_value = self.s.on_chip_binning.v

            # Set the new value:
            s.on_chip_binning.__set__(set_to)

            # t_exp correction:
            s.sc1.t_exp.__set__(s.sc1.t_exp.v * (set_to/old_value) ** (-2))
            s.sc2.t_exp.__set__(s.sc2.t_exp.v * (set_to / old_value) ** (-2))

            print((set_to//old_value)**2)

        # CONNECTIONS: Enable On-Chip binning on the camera
        self.radioButton_onchip_1.clicked.connect(lambda: partial(radioButton_onchip_clicked, self.s,  1)())
        self.radioButton_onchip_2.clicked.connect(lambda: partial(radioButton_onchip_clicked, self.s,  2)())
        self.radioButton_onchip_4.clicked.connect(lambda: partial(radioButton_onchip_clicked, self.s,  4)())

        # CONNECTION: "Use windowing" Checkbox
        self.checkBox_use_windowing.toggled.connect(lambda x: self.s.use_windowing.__set__(x))

        # CONNECTION: "Skip eval" button:
        self.checkBox_skip_eval.toggled.connect(lambda x: self.s.skip_eval.__set__(x))

        # SLOT: Save logic. Create new save folders if required
        def toggle_save_parameters(x):
            if not os.path.isdir(self.s.subfolder.v):
                os.mkdir(self.s.subfolder.v)
            s.save_parameters.__set__(x)

        # CONNECTION: Save logic. Create new save folders if required
        self.checkBox_save_parameters.clicked.connect(toggle_save_parameters)

        # SLOT: Save images?
        def toggle_save_raw_images():
            if not os.path.isdir(self.s.subfolder.v):
                os.mkdir(self.s.subfolder.v)
            s.save_raw_images.__set__(not s.save_raw_images.v)

        # SLOT: How many images to save before stopping?
        def toggle_save_amount():

            N = self.spinBox_numberSaveImages.value()

            print("toggle_save_amount thinks that is should set the final save index to: {} + {}".format(self.s.index.v, N))
            self.s.resetSaveAfter.__set__(self.s.index.v + N)
            self.checkBox_save_raw_images.setChecked(True)

        # CONNECTIONS: Save buttons
        self.checkBox_save_raw_images.toggled.connect(toggle_save_raw_images)
        self.checkBox_save_cam1.toggled.connect(lambda x: self.s.sc1.save.__set__(x))
        self.checkBox_save_cam2.toggled.connect(lambda x: self.s.sc2.save.__set__(x))
        self.spinBox_numberSaveImages.editingFinished.connect(partial(toggle_save_amount))

        # CONNECTIONS: Exposure spinbox implementation:
        self.doubleSpinBox_texp1.valueChanged.connect(lambda x: self.s.sc1.t_exp.__set__(x*1000))
        self.doubleSpinBox_texp2.valueChanged.connect(lambda x: self.s.sc2.t_exp.__set__(x*1000))

        # CONNECTIONS: Auto-Exposure settings
        self.checkBox_auto_exposure.toggled.connect(lambda x: self.s.auto_exposure.__set__(x))
        self.checkBox_use_AE_ROI.toggled.connect(lambda x: self.s.use_AE_ROI.__set__(x))
        self.checkBox_same_texp.toggled.connect(lambda x: self.s.texp_lock.__set__(x))

        # CONNECTION: Automatically find matching pairs of tbin_sub1 and tbin_sub2 based on exposure time
        self.checkBox_n1n2.toggled.connect(lambda x: self.s.auto_n1n2.__set__(x))

        # CONNECTIONS: "tbin-sub" spinbox implementation:
        self.spinBox_n1.valueChanged.connect(lambda x: self.s.sc1.tbin_sub.__set__(x))
        self.spinBox_n2.valueChanged.connect(lambda x: self.s.sc2.tbin_sub.__set__(x))

        # CONNECTIONS: Measure mode button:
        self.pushButton_measure_mode.clicked.connect(lambda x: s.measure_mode.__set__(True))
        self.pushButton_measure_mode.clicked.connect(lambda x: s.read_mode.__set__(False))

        # CONNECTIONS: Trigger button (Software <-> Hardware) control:
        self.radioButton_trigger_software.clicked.connect(lambda x: s.trigger_source.__set__("Software"))
        self.radioButton_trigger_software.clicked.connect(partial(set_trigger_source, self.s.sc2.cam, TriggerType.SOFTWARE))
        self.radioButton_trigger_hardware.clicked.connect(lambda x: s.trigger_source.__set__("Line0"))
        self.radioButton_trigger_hardware.clicked.connect(partial(set_trigger_source, self.s.sc2.cam, TriggerType.HARDWARE))

        # SLOT: Functionality of the "Find parallax" button:
        def button_queue_parallax_search_clicked():
            """
            Initializes a ParallaxThread to find a warp matrix for image1 and image2 and writes the warp matrix to the respective field of the
            Settings instance.
            :return: None
            """
            self.label_parallax_queued.setText("Waiting for thread...")

            # Extract data from ROIs:
            im1 = self.parallax_ROI(0)
            im2 = self.parallax_ROI(1)

            # Initialize a thread:
            parallax_thread = ParallaxThread(im1, im2, self.s)

            im1 = self.graphicsView.getImageItem().image
            im2 = self.graphicsView_2.getImageItem().image

            # Initialize a stitching thread:
            stitching_thread = StitchingThread(im1, im2, self.s)

            # sig emits the matrix, connect it with the entry in s
            parallax_thread.sig.connect(lambda x: self.s.warp_matrix.__set__(x))
            parallax_thread.sig.connect(lambda x: self.label_parallax_known.setText(str(np.round(x, 2)) +
                                        "\n phi = {} degrees".format(
                                            round(np.arcsin(x[0, 1]) * 180 / np.pi, 4))))

            parallax_thread.sig.connect(lambda x: print(str(np.round(x, 2)) +
                                        "\n phi = {} degrees".format(
                                            round(np.arcsin(x[0, 1]) * 180 / np.pi, 4))))

            parallax_thread.idlesig.connect(lambda x: self.label_parallax_queued.setText(x))

            stitching_thread.sig.connect(lambda x: self.s.stitch_M.__set__(x))

            # If in read mode, finishing the matrix search should call peakview():
            if self.s.read_mode.v:
                parallax_thread.sig.connect(partial(peak_view, self))

            parallax_thread.start()
            stitching_thread.start()

            while not parallax_thread.isFinished():
                QtWidgets.QApplication.processEvents()

        # CONNECTION: Start calculation of a warp matrix.
        self.button_queue_parallax.clicked.connect(button_queue_parallax_search_clicked)

        # CONNECTIONS: Functionality of the GUI elements that let the user edit the warp matrix manually
        self.pushButton_Lrotate.clicked.connect(lambda: s.warp_matrix.__set__(increase_angle(s.warp_matrix.v, 0.025, self)))
        self.pushButton_Rrotate.clicked.connect(lambda: s.warp_matrix.__set__(increase_angle(s.warp_matrix.v, -0.025, self)))

        self.pushButton_Tleft.clicked.connect(lambda: s.warp_matrix.__set__(increase_translation_x(s.warp_matrix.v, self.doubleSpinBox_translation_unit.value(), self)))
        self.pushButton_Tright.clicked.connect(lambda: s.warp_matrix.__set__(increase_translation_x(s.warp_matrix.v, -self.doubleSpinBox_translation_unit.value(), self)))
        self.pushButton_Tdown.clicked.connect(lambda: s.warp_matrix.__set__(increase_translation_y(s.warp_matrix.v, -self.doubleSpinBox_translation_unit.value(), self)))
        self.pushButton_Tup.clicked.connect(lambda: s.warp_matrix.__set__(increase_translation_y(s.warp_matrix.v, self.doubleSpinBox_translation_unit.value(), self)))

        self.pushButton_identity_matrix.clicked.connect(lambda: s.warp_matrix.__set__(np.eye(2, 3)))
        self.pushButton_identity_matrix.clicked.connect(lambda: self.label_parallax_known.setText(str(np.eye(2, 3))))

        # CONNECTIONS: Connect the image index slider with the text boxes next to it:
        self.horizontalSlider_stack_upper.valueChanged.connect(
            lambda: self.lineEdit_stack_upper.setText(str(self.horizontalSlider_stack_upper.value())))

        self.lineEdit_stack_upper.editingFinished.connect(
            lambda: self.horizontalSlider_stack_upper.setValue(int(self.lineEdit_stack_upper.text())))

        """Finally, load the last exposure time settings:
        When the GUI is loaded, read in the last exposure times of the cameras (They are still set in the cameras' nodemaps)"""
        if not self.s.read_mode.v:
            self.s.sc1.t_exp.v = get_node(self.s.sc1.cam, "ExposureTime", 2)
            self.s.sc2.t_exp.v = get_node(self.s.sc2.cam, "ExposureTime", 2)

            # Set the GUI elements accordingly
            self.doubleSpinBox_texp1.setValue(min(s.sc1.t_exp.v // 1000, 1000*1000))
            self.doubleSpinBox_texp2.setValue(min(s.sc2.t_exp.v // 1000, 1000*1000))

    """Exclusive to Read-Mode: Some elements should invoke refreshing the image view. This is done using the 'peak_view' function.
    Define the elements, whose state change refreshes the view:"""
    def connect_peak_view_in_readmode(self):
        """
        Connects GUI elements of the MainWindow to the peak_view function
        :return: None
        """

        self.checkBox_dark_correction.clicked.connect(partial(peak_view, self))
        self.checkBox_parallax_correction.clicked.connect(partial(peak_view, self))

        self.checkBox_ffc.clicked.connect(partial(peak_view, self))
        self.checkBox_dark_correction.clicked.connect(partial(peak_view, self))
        self.checkBox_NLC.clicked.connect(partial(peak_view, self))
        self.listWidget_ffc1.itemClicked.connect(partial(peak_view, self))
        self.listWidget_ffc2.itemClicked.connect(partial(peak_view, self))

        self.checkBox_background_correction.clicked.connect(partial(peak_view, self))
        self.checkBox_show_processed.clicked.connect(partial(peak_view, self))
        self.checkBox_auto_histogram.clicked.connect(partial(peak_view, self))
        self.checkBox_ignore_ROI_signals.toggled.connect(partial(peak_view, self))

        self.checkBox_plot_ROI_source_0.clicked.connect(partial(peak_view, self))
        self.checkBox_plot_ROI_source_1.clicked.connect(partial(peak_view, self))
        self.checkBox_plot_ROI_source_2.clicked.connect(partial(peak_view, self))
        self.spinBox_rolling_mean.valueChanged.connect(partial(peak_view, self))
        self.checkBox_grid.clicked.connect(partial(peak_view, self))

        self.pushButton_identity_matrix.clicked.connect(partial(peak_view, self))

        self.pushButton_Lrotate.clicked.connect(partial(peak_view, self))
        self.pushButton_Rrotate.clicked.connect(partial(peak_view, self))

        self.pushButton_Tleft.clicked.connect(partial(peak_view, self))
        self.pushButton_Tright.clicked.connect(partial(peak_view, self))
        self.pushButton_Tup.clicked.connect(partial(peak_view, self))
        self.pushButton_Tdown.clicked.connect(partial(peak_view, self))

        self.checkBox_offset_correction.clicked.connect(partial(peak_view, self))
        self.radio_background_manual.clicked.connect(partial(peak_view, self))
        self.radio_background_ROI.clicked.connect(partial(peak_view, self))
        self.lineEdit_offset.textChanged.connect(partial(peak_view, self))
        self.spinBox_tbin.editingFinished.connect(partial(peak_view, self))
        self.radioButton_units_1.clicked.connect(partial(peak_view, self))
        self.radioButton_units_3.clicked.connect(partial(peak_view, self))
        self.horizontalSlider_stack_upper.valueChanged.connect(partial(peak_view, self))
        self.spinBox_sbin_x.valueChanged.connect(partial(peak_view, self))
        self.spinBox_sbin_y.valueChanged.connect(partial(peak_view, self))

        for roi in self.offset_ROI.rois + self.avg_ROI.rois:
            roi.sigRegionChangeFinished.connect(partial(peak_view, self))

    def update_from_settings_instance(self, s):
        """
        Updates the GUI's parameter table based on the Settings instance
        :param s: Settings instance
        :return: None
        """

        # Update the non-cam entries:

        # Pull the names of parameters to display
        parameter_names = [parameter_name for parameter_name in s.__dir__()
                           if type(s.__getattribute__(parameter_name)) is SettingsEntry
                           and s.__getattribute__(parameter_name).show]

        # Set the header names based on the parameter names
        self.tableWidget.setVerticalHeaderLabels([s.__getattribute__(parameter_name).display_name
                                                  for parameter_name in parameter_names])

        # Set row- and column count based on the amount of parameter names
        self.tableWidget.setRowCount(len(parameter_names))
        self.tableWidget.setColumnCount(1)

        # For each parameter name, find the corresponding value
        for i in range(len(parameter_names)):

            parameter_name = parameter_names[i]
            parameter_value = str(s.__getattribute__(parameter_name))

            # Write the entry to the table
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(parameter_value))

        # Update the cam-specific entries:

        # Find all the CamSettings instances in the Settings instance
        cam_setting_instances = [s.__getattribute__(parameter_name) for parameter_name in s.__dir__()
                                 if type(s.__getattribute__(parameter_name)) is CamSettingsNew]

        # Pull the names of parameters to display
        parameter_names = [parameter_name for parameter_name in cam_setting_instances[0].__dir__()
                           if type(cam_setting_instances[0].__getattribute__(parameter_name)) is SettingsEntry
                           and cam_setting_instances[0].__getattribute__(parameter_name).show]

        # Set the header names based on the parameter names
        self.tableWidget_cameras.setVerticalHeaderLabels(
            [cam_setting_instances[0].__getattribute__(parameter_name).display_name
             for parameter_name in parameter_names])

        # Set row- and column count based on the amount of parameter names
        self.tableWidget_cameras.setRowCount(len(parameter_names))
        self.tableWidget_cameras.setColumnCount(len(cam_setting_instances))

        # For each parameter name, find the corresponding value
        for j in range(len(cam_setting_instances)):

            sc = cam_setting_instances[j]

            for i in range(len(parameter_names)):
                parameter_name = parameter_names[i]
                parameter_value = str(sc.__getattribute__(parameter_name))

                # Write the entry to the table
                self.tableWidget_cameras.setItem(i, j, QtWidgets.QTableWidgetItem(parameter_value))
