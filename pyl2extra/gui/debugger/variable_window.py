#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Nicu Tofan <nicu.tofan@gmail.com>
"""

import numpy

import PyQt4
from PyQt4 import QtGui, QtCore

import pyqtgraph as pg

#from theano.tensor.sharedvar import TensorSharedVariable
from theano.gof import Variable

from .gui import center
from .image import nparrayToQPixmap, gray2qimage

import logging
logger = logging.getLogger(__name__)

class VariableWindow(QtGui.QWidget):
    """
    Display the content of a Theano variable.
    """
    def __init__(self, mw, variable):
        """
        Constructor
        """
        super(VariableWindow, self).__init__()

        self.mw = mw
        self.variable = variable
        self.img_width = None
        self.img_height = None
        self.image_label = None
        self.image_widget = None
        self.init_ui()

        self.show_variable(self.variable)

    def init_ui(self):
        """
        Prepares the GUI.
        """

        self.resize(800, 600)
        self.setWindowTitle("Variable " + str(self.variable.__class__))
        center(self)

        self.splitter = QtGui.QSplitter()

        txt = QtGui.QTextEdit()
        txt.setLineWrapMode(QtGui.QTextEdit.NoWrap)
        txt.setReadOnly(True)
        txt.setStyleSheet('font: 8pt "Courier";')

        self.splitter.addWidget(txt)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)

        self.txt = txt
        self.grid = QtGui.QHBoxLayout()
        self.grid.setSpacing(1)
        self.grid.addWidget(self.splitter)
        self.setLayout(self.grid)

    def data_format(self, data_ty):
        """
        Get the format to use based on the name of the type.
        """
        if not isinstance(data_ty, str):
            data_ty = str(data_ty)
        if data_ty.startswith('float'):
            result = '%8.6f '
        elif 'int' in data_ty:
            result = '%4d '
        else:
            result = '%s '
        return result

    def show_1darray(self, variable):
        """
        Present an 1D array
        """
        assert isinstance(variable, numpy.ndarray)
        assert len(variable.shape) == 1
        val_frm = self.data_format(variable.dtype)

        r = 0
        line_txt = '    0: '
        for c in range(variable.shape[0]):
            line_txt = line_txt + val_frm % self.variable[c]
            r = r + 1
            if r >= 10:
                self.txt.append(line_txt)
                line_txt = '%5d: ' % r*10
                r = 0
        self.txt.append(line_txt)

    def print_2darray(self, variable):
        """
        Present an 2D array
        """
        assert isinstance(variable, numpy.ndarray)
        assert len(variable.shape) == 2
        val_frm = self.data_format(variable.dtype)

        line_txt = '    0: '
        for r in range(variable.shape[0]):
            line_txt = '%5d: ' % r
            for c in range(variable.shape[1]):
                line_txt = line_txt + val_frm % variable[r][c]
            self.txt.append(line_txt)

    def show_2darray(self, variable):
        """
        Present an 2D array
        """
        self.print_2darray(variable)
        self.add_image_gui(variable)

    def show_3darray(self, variable):
        """
        Present an 3D array
        """
        assert isinstance(variable, numpy.ndarray)
        assert len(variable.shape) == 3

        self.print_nd_one_layer(variable)
        self.add_image_gui(variable)

    def show_xdarray(self, variable):
        """
        Present an XD array
        """
        assert isinstance(variable, numpy.ndarray)
        assert len(variable.shape) >= 3

        self.print_nd_one_layer(variable)
        self.combos = []
        grid = QtGui.QVBoxLayout()
        ordered = []
        for i in range(len(variable.shape)):
            w = QtGui.QComboBox()
            w.addItem('on-screen')
            for j in range(variable.shape[i]):
                w.addItem(str(j))
            grid.addWidget(w)
            self.combos.append(w)
            if i > 1:
                w.setCurrentIndex(1)
                ordered.append(0)
            else:
                ordered.append(-1)
            self.connect(w, QtCore.SIGNAL('activated(int)'),
                         self.change_ndarray_layout)
            w.setSizePolicy(QtGui.QSizePolicy.Maximum,
                            QtGui.QSizePolicy.Fixed)
        self.lbl_ndarray_layout = QtGui.QLabel('')
        self.lbl_ndarray_layout.setWordWrap(True)
        grid.addWidget(self.lbl_ndarray_layout)
        spacer1 = QtGui.QSpacerItem(20, 40,
                                    QtGui.QSizePolicy.Fixed,
                                    QtGui.QSizePolicy.Expanding)
        grid.addSpacerItem(spacer1)                        
        self.grid.addLayout(grid)
        self.add_image_gui(variable)
        self.show_ndarray_layout((0,1), ordered)
        
    def show_ndarray_layout(self, onscreen, ordered):
        
        dims_shown = len(ordered) - 2
        tmp_var = self.variable.copy()
        tmp_var = tmp_var.swapaxes(onscreen[0], len(ordered) - 2)
        tmp_var = tmp_var.swapaxes(onscreen[1], len(ordered) - 1)
        ordered[onscreen[0]] = ordered[len(ordered) - 2]
        ordered[onscreen[1]] = ordered[len(ordered) - 1]
        
        for i in range(dims_shown):
            tmp_var = tmp_var[ordered[i]]
        tmp_swapped = tmp_var.swapaxes(0,1)

        self.image_widget.setImage(tmp_swapped)
        self.lbl_ndarray_layout.setText(str(tmp_var.shape))
        if tmp_var.shape[0] < 100 and tmp_var.shape[1] < 100:
            self.image_widget.setToolTip(str(tmp_var))
        else:
            self.image_widget.setToolTip('Array too large to display')

    def change_ndarray_layout(self, index):
        """
        """
        first_shown_dim = -1
        second_shown_dim = -1
        ordered = []
        for i, w in zip(range(len(self.combos)), self.combos):
            idx = w.currentIndex() - 1
            ordered.append(idx-1)
            if idx == -1:
                if first_shown_dim == -1:
                    first_shown_dim = i
                elif second_shown_dim == -1:
                    second_shown_dim = i
                else:  
                    self.lbl_ndarray_layout.setText('Only two dimensions can be on screen at a time')
                    return
        if second_shown_dim == -1:
            self.lbl_ndarray_layout.setText('Two dimensions must be on screen')
        else:
            self.lbl_ndarray_layout.setText('')
            self.show_ndarray_layout((first_shown_dim, second_shown_dim), ordered)
        
        
    def show_ndarray(self, variable):
        """
        """
        assert isinstance(variable, numpy.ndarray)
        dims = len(variable.shape)
        if dims == 1:
            self.show_1darray(variable)
        elif dims == 2:
            self.show_2darray(variable)
        elif dims == 3:
            b_image = False
            if variable.shape[0] == 3:
                b_image = True
                variable.swapaxes(0,2)
            elif variable.shape[1] == 3:
                b_image = True
                variable.swapaxes(1,2)
            elif variable.shape[2] == 3:
                b_image = True
            if b_image:
                self.show_3darray(variable)
            else:
                self.show_xdarray(variable)
        else:
            self.show_xdarray(variable)

    def show_list_like(self, variable):
        for i in variable:
            self.txt.append(str(i))

    def show_dict_like(self, variable):
        for i in variable:
            self.txt.append('%s: %s' % (str(i), str(variable[i])))

    def show_variable(self, variable):
        if isinstance(variable, (int, float, long, bool, str, unicode)):
            self.txt.append(str(variable))
        elif isinstance(variable, Variable):
            try:
                self.variable = variable.eval()
                self.show_variable(self.variable)
            except Exception, exc:
                self.txt.append(str(exc))
        elif isinstance(variable, numpy.ndarray):
            self.show_ndarray(variable)
        elif isinstance(variable, list):
            self.show_list_like(variable)
        elif isinstance(variable, dict):
            self.show_dict_like(variable)
        elif isinstance(variable, tuple):
            if len(variable) < 10:
                self.txt.append(str(variable))
            else:
                self.show_list_like(variable)


    def add_image_gui(self, variable):




        b_label = False
        if len(variable.shape) == 3:
            if variable.shape[0] == 3:
                b_label = True
                variable.swapaxes(0,2)
            elif variable.shape[1] == 3:
                b_label = True
                variable.swapaxes(1,2)
            elif variable.shape[2] == 3:
                b_label = True

        if b_label:
            grid1 = QtGui.QVBoxLayout()
            grid1.setSpacing(10)
            spacer1 = QtGui.QSpacerItem(20, 40,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Minimum)

            spacer2 = QtGui.QSpacerItem(20, 40,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Minimum)          
            
            scaled_w = 400
            scaled_h = scaled_w * variable.shape[1] / variable.shape[0]
            to_view = nparrayToQPixmap(variable)

            widget = QtGui.QLabel(self)
            widget.resize(scaled_w, scaled_h)
            widget.setMinimumSize(scaled_w, scaled_h)
            widget.setMaximumSize(scaled_w, scaled_h)
            widget.setScaledContents(True)
            widget.setPixmap(to_view)
            grid1.addSpacerItem(spacer1)
            grid1.addWidget(widget)
            grid1.addSpacerItem(spacer2)    
            
            container = QtGui.QWidget()
            container.setLayout(grid1)
            
            self.splitter.addWidget(container)
        else:
            widget = pg.ImageView()
            widget.setImage(variable)
            self.splitter.addWidget(widget)

        self.image_widget = widget
        #self.grid.addLayout(grid1)

    def _add_image_gui(self):
        """
        Inserts the image components into the main grid.
        """
        if self.img_height > 4000 or self.img_width > 4000:
            return

        grid1 = QtGui.QVBoxLayout()
        grid1.setSpacing(10)

        self.image_label = QtGui.QLabel(self)
        scaled_w = 400
        scaled_h = scaled_w * self.img_height / self.img_width
        self.image_label.resize(scaled_w, scaled_h)
        self.image_label.setMinimumSize(scaled_w, scaled_h)
        self.image_label.setScaledContents(True)

        spacer = QtGui.QSpacerItem(20, 40,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Minimum)
        grid1.addSpacerItem(spacer)
        grid1.addWidget(self.image_label)
        spacer = QtGui.QSpacerItem(20, 40,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Minimum)
        grid1.addSpacerItem(spacer)

        self.grid.addLayout(grid1)

    def print_nd_one_layer(self, var, original=None):
        """
        Recursive function to print matrices with higher dimensions.
        """
        if original is None:
            original = str(var.shape)

        if len(var.shape) == 2:
            self.print_2darray(var)
        else:
            for i in range(var.shape[0]):
                self.txt.append('='*80)
                self.txt.append('  %d  - %s (original %s)' % (i, str(var.shape),
                                                              original))
                self.txt.append('='*80)
                self.print_nd_one_layer(var[i, :], original)

    def old_init_ui(self):
        txt = QtGui.QTextEdit()
        def flat2D():
            """
            Prints a 2D matrix in text area.
            """
            self.img_width = self.variable.shape[0]
            self.img_height = self.variable.shape[1]
            self.add_image_gui()

            for r in range(self.img_height):
                line_txt = ''
                for c in range(self.img_width):
                    line_txt = line_txt + data_format % self.variable[c, r]
                txt.append(line_txt)

                to_view = gray2qimage(self.variable)
                self.image_label.setPixmap(to_view)

        if isinstance(self.variable, numpy.ndarray):
            data_format = self.data_format(str(self.variable.dtype))
            b_has_been_shown = False
            if len(self.variable.shape) == 1:
                r = 0
                line_txt = ''
                for c in range(self.variable.shape[0]):
                    line_txt = line_txt + data_format % self.variable[c]
                    r = r + 1
                    if r >= 10:
                        txt.append(line_txt)
                        line_txt = ''
                        r = 0
                txt.append(line_txt)
            elif len(self.variable.shape) == 2:
                # grey
                flat2D()
                b_has_been_shown = True
            elif len(self.variable.shape) == 333:
                if self.variable.shape[2] == 1:
                    # grey
                    self.variable = numpy.reshape(self.variable,
                                                  self.variable.shape[0],
                                                  self.variable.shape[1])
                    flat2D()
                    b_has_been_shown = True
                elif self.variable.shape[2] == 3:
                    # color

                    self.img_width = self.variable.shape[0]
                    self.img_height = self.variable.shape[1]
                    self.add_image_gui()

                    to_view = nparrayToQPixmap(self.variable)
                    self.image_label.setPixmap(to_view)

                    for r in range(self.img_height):
                        line_txt = ''
                        for c in range(self.img_width):
                            line_txt = line_txt + '['
                            for i in (0, 1, 2):
                                line_txt = line_txt + data_format % self.variable[r, c, i]
                            line_txt = line_txt + ']'
                        txt.append(line_txt)
                    b_has_been_shown = True

            def show_one_layer(var):
                """
                Recursive function to print matrices with higher dimensions.
                """
                if len(var.shape) == 2:
                    for r in range(var.shape[0]):
                        line_txt = ''
                        for c in range(var.shape[1]):
                            line_txt = line_txt + data_format % var[r, c]
                        txt.append(line_txt)
                    txt.append('')
                else:
                    for i in range(var.shape[0]):
                        txt.append('========================')
                        txt.append('  %d  - %s' % (i, str(var.shape)))
                        txt.append('========================')
                        show_one_layer(var[i, :])

            if not b_has_been_shown:
                show_one_layer(self.variable)
                b_has_been_shown = True

        elif isinstance(self.variable, Variable):
            txt.append(str(self.variable.eval()))
        else:
            for i in dir(self.variable):
                txt.append(i)


