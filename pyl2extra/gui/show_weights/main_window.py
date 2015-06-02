#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main window for navigating the Pylearn2 model app.

Don't run this script directly; instead, use
``pyl2extra/scripts/show_weights.py``.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"


import logging
import numpy
import os
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import QSettings
from pylearn2.utils import serial, safe_zip
from pylearn2.models.model import Model
import pyqtgraph as pg
import theano
import sys

from pyl2extra.gui.guihelpers import center
from pyl2extra.gui.guihelpers import make_act
from pyl2extra.gui.show_weights.all_weights_widget import AllWeightsWidget

logger = logging.getLogger(__name__)
Q = QtGui.QMessageBox.question


class MainWindow(QtGui.QMainWindow):
    """
    Main application window.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.settings = QSettings('pyl2extra', 'show_weights')
        self.init_actions()
        self.init_ui()
        theano.config.experimental.unpickle_gpu_on_cpu = True
        self.variable = None
        self.all_wwidget = []
        self.setWindowTitle('Model Browser')

    def init_ui(self):
        """
        Prepare GUI for main window.
        """
        try:
            point = self.settings.value('mw/pos', type=QtCore.QPoint)
            self.move(point)
        except TypeError:
            center(self)

        try:
            size = self.settings.value('mw/size', type=QtCore.QSize)
            self.resize(size)
        except TypeError:
            self.resize(900, 800)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle('Browse model')

        gridw = QtGui.QWidget()
        grid = QtGui.QHBoxLayout()
        grid.setSpacing(15)

        self.lv_top = QtGui.QTreeWidget(self)
        self.lv_top.setMaximumWidth(150)
        self.lv_top.setHeaderLabel('Content')
        self.lv_top.setToolTip('''The list of parameters.
The list is extracted when a file is loaded.
The user can select a parameter to see information
about it and a graphical representation.
        ''')
        self.lv_top.setWhatsThis('The list of parameters')
        sgn = 'currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)'
        self.connect(self.lv_top, QtCore.SIGNAL(sgn),
                     self.current_item_changed)
        self.lv_top.itemDoubleClicked.connect(self.double_click_param)
        grid.addWidget(self.lv_top)

        self.info_grid = QtGui.QVBoxLayout()
        self.lbl_info = QtGui.QLabel('Information')
        self.lbl_info.setWhatsThis('Name of the selected parameter')
        self.info_grid.addWidget(self.lbl_info)
        self.lbl_type = QtGui.QLabel('Type: ')
        self.lbl_type.setToolTip('Type can only be ndarray or CudaNdarray '
                                 'at this time')
        self.lbl_type.setWhatsThis(self.lbl_type.toolTip())
        self.info_grid.addWidget(self.lbl_type)
        self.lbl_shape = QtGui.QLabel('Shape: ')
        self.lbl_type.setToolTip('Shape for selected parameter')
        self.lbl_type.setWhatsThis(self.lbl_type.toolTip())
        self.info_grid.addWidget(self.lbl_shape)
        self.lbl_value = QtGui.QLabel('')
        self.lbl_type.setToolTip('Value, shape for display or warnings')
        self.lbl_type.setWhatsThis(self.lbl_type.toolTip())
        self.info_grid.addWidget(self.lbl_value)

        self.combos = []
        spat = QtGui.QSpacerItem(10, 10,
                                 QtGui.QSizePolicy.Minimum,
                                 QtGui.QSizePolicy.Expanding)
        self.info_grid.addSpacerItem(spat)
        grid.addLayout(self.info_grid)


        self.img_crt = pg.ImageView()
        self.img_crt.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)

        grid.addWidget(self.img_crt)

        self.plt_crt = pg.PlotWidget()
        self.plt_crt.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        self.v_line = None
        self.h_line = None
        # self.proxy = pg.SignalProxy(self.plt_crt.scene().sigMouseMoved,
        #                            rateLimit=60, slot=self.plot_mouse_moved)
        grid.addWidget(self.plt_crt)
        self.plt_crt.hide()

        gridw.setLayout(grid)
        self.setCentralWidget(gridw)

    def double_click_param(self, item, column):
        """
        An item in the main list has been double-clicked.
        """
        ex = AllWeightsWidget(item.parv)
        ex.show()
        self.all_wwidget.append(ex)
        ex.destroyed.connect(self._aww_closed)

    def _aww_closed(self):
        """
        An item in the main list has been double-clicked.
        """
        self.all_wwidget.remove(self.sender())

    def plot_mouse_moved(self, evt):
        """
        Mouse moving in the plot (not working right now).
        TODO: fixit
        """
        raise NotImplementedError()
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        vb = self.img_crt.getView()
        mouse_point = vb.mapSceneToView(pos)
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            index = int(mouse_point.x())
            if index > 0 and index < len(data1):
                label.setText("<span style='font-size: 12pt'>x=%0.1f,"
                              "   <span style='color: red'>y1=%0.1f</span>,"
                              "   <span style='color: green'>y2=%0.1f</span>"
                              % (mouse_point.x(), data1[index], data2[index]))
        if not self.v_line is None:
            self.v_line.setPos(mouse_point.x())
            self.h_line.setPos(mouse_point.y())

    def combo_xdarray(self, var_shape):
        """
        Add combo boxes for dimensions of a ndarray.
        """
        self.combos = []
        ordered = []
        for i in range(len(var_shape)):
            wcombo = QtGui.QComboBox()
            wcombo.addItem('on-screen')
            for j in range(var_shape[i]):
                wcombo.addItem(str(j))
            self.info_grid.addWidget(wcombo)
            self.combos.append(wcombo)
            if i > 1:
                wcombo.setCurrentIndex(1)
                ordered.append(0)
            else:
                ordered.append(-1)
            self.connect(wcombo, QtCore.SIGNAL('activated(int)'),
                         self.change_ndarray_layout)
            wcombo.setSizePolicy(QtGui.QSizePolicy.Maximum,
                            QtGui.QSizePolicy.Fixed)
        self.show_ndarray_layout((0, 1), ordered)

    def change_ndarray_layout(self, index):
        """
        Signal handler for combo boxes that are used to choose on-screen
        dimensions.

        Parameters
        ----------
        index : int
            Zero-based index of the selected element. First element (0)
            is always ``on-screen`` and it is followed by succesive integers.
        """
        first_shown_dim = -1
        second_shown_dim = -1
        ordered = []
        for i, w in enumerate(self.combos):
            idx = w.currentIndex() - 1
            ordered.append(idx)
            if idx == -1:
                if first_shown_dim == -1:
                    first_shown_dim = i
                elif second_shown_dim == -1:
                    second_shown_dim = i
                else:
                    self.lbl_value.setText('Only two dimensions can '
                                           'be on screen at a time')
                    return
        if second_shown_dim == -1:
            self.lbl_value.setText('Two dimensions must be on screen')
        else:
            self.lbl_value.setText('')
            self.show_ndarray_layout((first_shown_dim, second_shown_dim),
                                     ordered)

    def show_ndarray_layout(self, onscreen, ordered):
        """
        Adjusts the image component to display user selection.

        Parameters
        ----------
        onscreen : tuple or list of two ints
            Specifies the index of the dimensions that should be presented
            to the user.
        ordered : list of ints
            A list that has the same lenghts as the shape of current value.
            Each member indicates what dimension should be shown for that
            axis, with -1 for dimensions that are in `onscreen`.
        """
        if self.variable is None:
            return
        onscreen = list(onscreen)
        last_1 = len(ordered) - 1
        last_2 = len(ordered) - 2
        dims_shown = len(ordered) - 2
        tmp_var = self.variable.copy()
        if onscreen[0] != last_2:
            tmp_var = tmp_var.swapaxes(onscreen[0], last_2)
        if onscreen[1] == last_2:
            onscreen[1] = onscreen[0]
        ordered[onscreen[0]] = ordered[last_2]
        if onscreen[1] != last_1:
            tmp_var = tmp_var.swapaxes(onscreen[1], last_1)
        ordered[onscreen[1]] = ordered[last_1]

        for i in range(dims_shown):
            tmp_var = tmp_var[ordered[i]]
        tmp_swapped = tmp_var.swapaxes(0, 1)

        self.img_crt.setImage(tmp_swapped)
        self.lbl_value.setText('%dx%d shown' %
                               (tmp_var.shape[0], tmp_var.shape[1]))
        if tmp_var.shape[0] < 20 and tmp_var.shape[1] < 20:
            self.img_crt.setToolTip(str(tmp_var))
        else:
            self.img_crt.setToolTip('Array too large to display')

    def init_actions(self):
        """
        Prepares the actions, then menus & toolbars.
        """

        self.act_exit = make_act('&Exit', self,
                                 'door_in.png',
                                 'Ctrl+Q',
                                 'Exit application',
                                 self.close)
        self.act_help = make_act('&Quick intro...', self,
                                 'question.png',
                                 'Ctrl+H',
                                 'Guidance in using the application',
                                 self.show_help)
        self.act_load_mdl = make_act('&Open pickled model...', self,
                                     'folder_database.png',
                                     'Ctrl+O',
                                     'Load file and preprocess',
                                     self.browse_model)
        self.act_export_img = make_act('&Export value as image...', self,
                                       'folder_database.png',
                                       'Ctrl+I',
                                       'Creates a single image for a '
                                       'multi-dimensional variable',
                                       self.export_img)

        menubar = self.menuBar()

        menu_file = menubar.addMenu('&File')
        menu_file.addAction(self.act_load_mdl)
        menu_file.addAction(self.act_exit)

        menu_help = menubar.addMenu('&Help')
        menu_help.addAction(self.act_help)

        self.toolbar = self.addToolBar('General')
        self.toolbar.addAction(self.act_load_mdl)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_exit)

    def closeEvent(self, event):
        """
        Build-in close event.
        """
        reply = Q(self, 'Closing...',
                  "Are you sure you want to quit?",
                  QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
                  QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            self.settings.setValue('mw/pos', self.pos())
            self.settings.setValue('mw/size', self.size())

            del self.settings
            event.accept()
            QtCore.QCoreApplication.exit(0)
        else:
            event.ignore()

    def clear_image_widget(self):
        """
        Clears GUI elements that change with selection in list of parameters..
        """
        self.img_crt.clear()
        plit = self.plt_crt.getPlotItem()
        plit.clear()
        for cmb in self.combos:
            cmb.deleteLater()
        self.combos = []

    def show_value_nd(self, value):
        """
        Presents the value in the image.

        Parameters
        ----------
        value : numpy.ndarray
            The array to show in the interface.
        """
        def set_plt_mode(label):
            """Show plot and hide image"""
            self.plt_crt.show()
            self.img_crt.hide()
            self.lbl_value.setText(label)

        def set_img_mode(label):
            """Show image and hide plot"""
            self.plt_crt.hide()
            self.img_crt.show()
            self.lbl_value.setText(label)

        if len(value.shape) == 0:
            set_img_mode(str(value))
        elif len(value.shape) == 1:
            set_plt_mode('')
            plit = self.plt_crt.getPlotItem()
            plit.setTitle('Data')
            self.v_line = pg.InfiniteLine(angle=90, movable=False)
            self.h_line = pg.InfiniteLine(angle=0, movable=False)
            self.plt_crt.addItem(self.v_line, ignoreBounds=True)
            self.plt_crt.addItem(self.h_line, ignoreBounds=True)
            plit.plot(x=range(value.shape[0]+1), y=value, stepMode=True)
            plit.setLabel('left', text='Value')
            plit.setLabel('bottom', text='index')
        elif len(value.shape) == 2:
            set_img_mode('')
            self.img_crt.setImage(value)
        else:
            set_img_mode('')
            self.combo_xdarray(value.shape)
            self.change_ndarray_layout(0)

    def show_value(self, value):
        """
        Presents the value in the image or in the plot.
        """
        #if isinstance(value, theano.sandbox.cuda.CudaNdarray):
        if isinstance(value.__class__, theano.sandbox.cuda.CudaNdarrayType):
            value = numpy.asarray(value)
            self.lbl_type.setText('Type: CudaNdarray')
            self.lbl_shape.setText('Shape: %s' % str(value.shape))
            self.show_value_nd(value)
        elif isinstance(value, numpy.ndarray):
            self.lbl_type.setText('Type: ndarray')
            self.lbl_shape.setText('Shape: %s' % str(value.shape))
            self.show_value_nd(value)
        else:
            self.lbl_type.setText('Type: unknown')
            self.lbl_shape.setText('Shape: -')

    def current_item_changed(self, current, previous):
        """
        The top level list's current item changed.
        """
        self.variable = current.parv
        self.clear_image_widget()
        if current is None:
            self.lbl_info.setText('Information')
            self.lbl_type.setText('Type: ')
            self.lbl_shape.setText('Shape: ')
            self.lbl_value.setText('')
        else:
            self.lbl_info.setText(current.text(0))
            # current.par = par
            self.show_value(current.parv)

    def unload_model(self):
        """
        Remove current model from ui.
        """
        self.lv_top.clear()
        self.clear_image_widget()

    def load_model(self, model):
        """
        Slot that loads a model object (not file).

        Parameters
        ----------
        model : Model
            The model to load.
        """
        try:
            logger.debug('Loading model %s', str(model))
            pras_list = model.get_params()
            parv_list = model.get_param_values()

            for par, parv in safe_zip(pras_list, parv_list):
                tvi = QtGui.QTreeWidgetItem()
                tvi.setText(0, par.name)
                tvi.par = par
                tvi.parv = parv
                self.lv_top.addTopLevelItem(tvi)

            logger.debug('Model loaded')
        except Exception, exc:
            logger.error('Loading image file failed', exc_info=True)
            QtGui.QMessageBox.warning(self, 'Exception', str(exc))

    def load_model_file(self, fname):
        """
        Read a model file and use `load_model()` to show it.

        Parameters
        ----------
        fname : str
            Path to model file.
        """
        model = serial.load(fname)
        if not isinstance(model, Model):
            QtGui.QMessageBox.warning(self, 'Error',
                                      'Expecting a model object; got a %s' %
                                      str(model.__class__))
        else:
            path, name = os.path.split(fname)
            self.settings.setValue('mw/model_path', path)
            self.setWindowTitle('[%s] - Visualize model' % name)
            self.load_model(model)

    def browse_model(self):
        """
        Slot that browse for and loads a model file.
        """
        path = self.settings.value('mw/model_path')
        if path is None:
            path = os.getcwd()

        fname = QtGui.QFileDialog.getOpenFileName(self,
                                                  'Open pickled model file',
                                                  path,
                                                  '*.pkl')
        if fname is None or len(fname) == 0:
            return
        self.load_model_file(fname)

    def export_img(self):
        """
        Create an image for values in an ndarray.
        """
        if self.value is None:
            QtGui.QMessageBox.warning(self, "Error", "No variable selected")
            return
        elif isinstance(self.value.__class__,
                        theano.sandbox.cuda.CudaNdarrayType):
            value = numpy.asarray(self.value)
        elif isinstance(self.value, numpy.ndarray):
            value = self.value
        else:
            QtGui.QMessageBox.warning(self, "Error",
                                    "Can't generate weights report from "
                                    "a %s instance" %
                                    str(self.value.__class__))
            return

        if len(value.shape) < 3:
            QtGui.QMessageBox.warning(self, "Error",
                                    "The variable must have at least three "
                                    "axes; current one has %d" %
                                    len(value.shape))
            return

        aww = AllWeightsWidget(value)
        aww.show()

    def show_help(self):
        """
        Present help screen.
        """
        QtGui.QMessageBox.information(self, 'Help for show_weights', """
The application allows you to show the weights of a class
that inherits from pylearn2's Model.

To start, goto File > Open pickled model... and select the
file that has your saved mode. Loading might take a while
and the window will look frozen while loading.

Once the model was loaded the parameters are presented in the
list on the left side. Select an entry to view the content for
that parameter.

For uni-dimensional values the program shows a simple,
histogram-like graph. For values with two dimensions and
more an image is presented.

If the value has more than two dimensions you can choose which
dimensions are shown using the combo-boxed that are shown in the
second vertical area next to list ov parameters.

The values shown in the image can be customized by adjusting
the treshold and chenging the way values are mapped to colors.
""")

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
