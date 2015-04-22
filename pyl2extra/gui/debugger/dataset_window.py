#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Nicu Tofan <nicu.tofan@gmail.com>
"""

import numpy
from PyQt4 import QtGui

from .gui import center
from .image import nparrayToQPixmap

class DatasetWindow(QtGui.QWidget):
    """
    Presents the content of a PyLearn2 Dataset.

    TODO: Actually only DenseDesignMatrix is supported right now.

    Parameters
    ----------
    dataset :  pylearn2.datasets.dataset.Dataset
        A dataset to show in the GUI.
    """
    def __init__(self, mw, dataset):
        """
        Constructor
        """

        super(DatasetWindow, self).__init__()
        
        self.mw = mw
        self.index = -1
        self.dataset = dataset
        self.topo_shape = self.dataset.get_batch_topo(1).shape
        self.channels = self.topo_shape[self.dataset.axes.index('c')]
        self.img_width = self.topo_shape[self.dataset.axes.index(0)]
        self.img_height = self.topo_shape[self.dataset.axes.index(1)]

        self.init_ui()
        if self.loc_spbx.maximum() > 0:
            self.change_image(1)

    def init_ui(self):
        """
        Prepares the GUI.
        """
        self.resize(800, 600)
        self.setWindowTitle(str(self.dataset.__class__))
        center(self)

        prev_btn = QtGui.QPushButton('Previous', self)
        next_btn = QtGui.QPushButton('Next', self)
        self.loc_spbx = QtGui.QSpinBox(self)
        self.loc_spbx.setMinimum(1)
        self.loc_spbx.setMaximum(self.dataset.get_num_examples())
        prev_btn.clicked.connect(self.btn_prev)
        next_btn.clicked.connect(self.btn_next)
        self.loc_spbx.valueChanged.connect(self.change_image)

        self.image_label = QtGui.QLabel(self)
        scaled_w = 600
        scaled_h = scaled_w * self.img_height / self.img_width
        self.image_label.resize(scaled_w, scaled_h)
        self.image_label.setMinimumSize(scaled_w, scaled_h)
        self.image_label.setScaledContents(True)

        tree_widget = QtGui.QTreeWidget()
        header = QtGui.QTreeWidgetItem(["Name", "Value"])
        tree_widget.setHeaderItem(header)
        self.load_attribs(tree_widget)

        grid1 = QtGui.QHBoxLayout()
        grid1.setSpacing(10)

        grid1.addWidget(prev_btn)
        grid1.addWidget(self.loc_spbx)
        grid1.addWidget(next_btn)

        grid2 = QtGui.QHBoxLayout()
        grid2.setSpacing(10)
        grid2.addWidget(self.image_label)
        grid2.addWidget(tree_widget)

        grid = QtGui.QVBoxLayout()
        grid.setSpacing(10)

        grid.addLayout(grid1)
        grid.addLayout(grid2)
        self.setLayout(grid)

    def show_optional_attr(self, tree_widget, s_name):
        """
        Looks for an attribute in thge dataset and shows it if present.
        """
        if hasattr(self.dataset, s_name):
            QtGui.QTreeWidgetItem(tree_widget,
                                  [s_name, str(getattr(self.dataset, s_name))])

    def load_attribs(self, tree_widget):
        """
        Load the attributes from undelying object.
        """
        QtGui.QTreeWidgetItem(tree_widget,
                              ["design shape", str(self.dataset.X.shape)])
        QtGui.QTreeWidgetItem(tree_widget,
                              ["topo shape", str(self.topo_shape)])
        QtGui.QTreeWidgetItem(tree_widget,
                              ["axes", str(self.dataset.axes)])
        QtGui.QTreeWidgetItem(tree_widget,
                              ["data_specs", str(self.dataset.data_specs)])

        if not self.dataset.y is None:
            QtGui.QTreeWidgetItem(tree_widget,
                                  ["y shape", str(self.dataset.y.shape)])

        self.show_optional_attr(tree_widget, 'SHAPE')
        self.show_optional_attr(tree_widget, 'TRAIN_PERCENT')
        self.show_optional_attr(tree_widget, 'EXCLUDED')
        self.show_optional_attr(tree_widget, 'which_set')
        self.show_optional_attr(tree_widget, 'which_feat')
        self.show_optional_attr(tree_widget, 'one_hot')
        self.show_optional_attr(tree_widget, 'ntrain')
        self.show_optional_attr(tree_widget, 'ntest')
        self.show_optional_attr(tree_widget, 'datapath')
        self.show_optional_attr(tree_widget, 'nexamples')
        self.show_optional_attr(tree_widget, 'center')
        self.show_optional_attr(tree_widget, 'rescale')

        if hasattr(self.dataset, 'label_names'):
            l_names = QtGui.QTreeWidgetItem(tree_widget, ['label_names'])
            for l_name in self.dataset.label_names:
                tvi = QtGui.QTreeWidgetItem(l_names)
                tvi.setText(0, l_name)
                tvi.setText(1, str(self.dataset.label_names[l_name]))

        # show values as tooltips
        for i in range(tree_widget.topLevelItemCount()):
            top_it = tree_widget.topLevelItem(i)
            top_it.setToolTip(0, top_it.text(1))
            top_it.setToolTip(1, top_it.text(1))

    def btn_prev(self):
        """
        Slot to get to previous image.
        """
        self.loc_spbx.setValue(self.loc_spbx.value() - 1)

    def btn_next(self):
        """
        Slot to get to next image.
        """
        self.loc_spbx.setValue(self.loc_spbx.value() + 1)

    def change_image(self, index):
        """
        Slot to change the image.
        """
        self.index = index
        index = index - 1

        reshaped = numpy.reshape(self.dataset.X[index],
                                 (1, self.dataset.X.shape[1]))
        to_view = self.dataset.get_topological_view(reshaped)
        if self.dataset.get_topo_batch_axis() != 0:
            to_view = numpy.swapaxes(to_view,
                                     self.dataset.axes.index('b'),
                                     self.dataset.axes.index('c'))
        assert to_view.shape[0] == 1
        to_view = numpy.reshape(to_view, (to_view.shape[1],
                                          to_view.shape[2],
                                          to_view.shape[3]))
        to_view = nparrayToQPixmap(to_view)
        self.image_label.setPixmap(to_view)
