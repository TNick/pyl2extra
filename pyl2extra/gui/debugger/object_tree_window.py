#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Nicu Tofan <nicu.tofan@gmail.com>
"""

from types import BuiltinFunctionType, TypeType, IntType, LongType, DictType
from types import FloatType, BooleanType, StringType, UnicodeType

import numpy
from PyQt4 import QtGui, QtCore

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.train import Train
from pylearn2.models.model import Model

from theano.tensor import Tensor
from theano.gof import Variable
from theano.gof.fg import MissingInputError

from pyl2extra.gui.guihelpers import center
from learn_spot.gui.utils import get_icon

import logging
logger = logging.getLogger(__name__)

interx = 0

STRINGABLE_TYPES = (IntType, LongType, FloatType, TypeType,
                    BooleanType, StringType, UnicodeType)
PYL2_TYPED = (Model, Train, Dataset)

class ObjectTreeWindow(QtGui.QWidget):
    """
    Presents the content of a PyLearn2 Dataset.

    TODO: Actually only DenseDesignMatrix is supported right now.

    Parameters
    ----------
    obj :  object
        The object to be presented.
    """
    def __init__(self, mw, obj_tree):
        """
        Constructor
        """
        super(ObjectTreeWindow, self).__init__()
        
        self.yaml_build_ins = False
        self.yaml_methods = False
        self.yaml_filter_hidden = True
        self.yaml_max_depth = 10
        
        self.mw = mw
        self.obj_tree = obj_tree
        self.tree_widget = None
        self.flat_object_list = []

        self.init_ui()
        self.refresh_object_tree()

    def init_ui(self):
        """
        Prepares the GUI.
        """
        self.resize(800, 600)
        center(self)

        self.grid = QtGui.QVBoxLayout()
        self.grid.setSpacing(10)

        caption = "Object Tree "
        if hasattr(self.obj_tree, 'YAML'):
            caption = caption + self.obj_tree.YAML
        self.setWindowTitle(caption)

        tree_widget = QtGui.QTreeWidget()
        header = QtGui.QTreeWidgetItem(["Name", "Type", "Value"])
        tree_widget.setHeaderItem(header)
        tree_widget.setAlternatingRowColors(True)

        # event handlers like custom menu, double click
        tree_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        tree_widget.customContextMenuRequested.connect(self.object_tree_contextual)
        tree_widget.itemDoubleClicked.connect(self.object_tree_double_click)
        
        btn_refresh = QtGui.QToolButton()
        btn_refresh.setIcon(get_icon('refresh_all.png'))
        btn_refresh.setToolTip('Reload the object')
        self.connect(btn_refresh, QtCore.SIGNAL('clicked()'),
                     self.refresh_object_tree)
        
        btn_methods = QtGui.QToolButton()
        #btn_methods.setIcon(get_icon('refresh_all.png'))
        btn_methods.setToolTip('Hide methods in tree')
        btn_methods.setCheckable(True)
        btn_methods.setCheckable(self.yaml_methods)
        self.connect(btn_methods, QtCore.SIGNAL('toggled(bool)'),
                     self.toggle_methods)
        
        btn_hidden = QtGui.QToolButton()
        #btn_hidden.setIcon(get_icon('refresh_all.png'))
        btn_methods.setCheckable(True)
        btn_methods.setCheckable(self.yaml_filter_hidden)
        btn_hidden.setToolTip('Hide hidden components in tree')
        self.connect(btn_hidden, QtCore.SIGNAL('toggled(bool)'),
                     self.toggle_hidden)
                     
        btn_bldins = QtGui.QToolButton()
        #btn_bldins.setIcon(get_icon('refresh_all.png'))
        btn_methods.setCheckable(True)
        btn_methods.setCheckable(self.yaml_build_ins)
        btn_bldins.setToolTip('Hide build-in components in tree')
        self.connect(btn_bldins, QtCore.SIGNAL('toggled(bool)'),
                     self.toggle_buildins)
        
        lbl_max_d = QtGui.QLabel('Max depth')

        sp_port_rcv = QtGui.QSpinBox()
        sp_port_rcv.setMinimum(1)
        sp_port_rcv.setMaximum(100)
        sp_port_rcv.setValue(self.yaml_max_depth)
        sp_port_rcv.setToolTip('Port for command and control.')
        self.connect(sp_port_rcv, QtCore.SIGNAL('valueChanged(int)'),
                     self.change_max_depth)
        
        spacer1 = QtGui.QSpacerItem(20, 40,
                                    QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Fixed)
        
        grid_btm = QtGui.QHBoxLayout()
        grid_btm.setSpacing(10)
        grid_btm.addSpacerItem(spacer1)
        grid_btm.addWidget(lbl_max_d)
        grid_btm.addWidget(sp_port_rcv)
        grid_btm.addWidget(btn_methods)
        grid_btm.addWidget(btn_hidden)
        grid_btm.addWidget(btn_bldins)
        grid_btm.addWidget(btn_refresh)

        self.grid.addWidget(tree_widget)
        self.grid.addLayout(grid_btm)
        self.setLayout(self.grid)
        self.tree_widget = tree_widget

    def change_max_depth(self, value):
        """
        Slot to change the maximum depth of the tree
        """
        self.yaml_max_depth = value
        self.refresh_object_tree()
        
    def toggle_buildins(self, value):
        """
        Slot to change buildins visibility
        """
        self.yaml_build_ins = value
        self.refresh_object_tree()
        
    def toggle_hidden(self, value):
        """
        Slot to change hidden symbols visibility
        """
        self.yaml_filter_hidden = value
        self.refresh_object_tree()
        
    def toggle_methods(self, value):
        """
        Slot to change methods visibility
        """
        self.yaml_methods = value
        self.refresh_object_tree()
        
    def object_tree_contextual(self, position):
        """
        Slot for contextual menu in the YAML tree view.
        """

        tree_widget = self.sender()
        indexes = tree_widget.selectedIndexes()
        if len(indexes) != 1:
            return

        menu = QtGui.QMenu(tree_widget)
        act_methods = menu.addAction('Methods')
        act_methods.setCheckable(True)
        act_methods.setChecked(self.yaml_methods)

        sel_act = menu.exec_(tree_widget.mapToGlobal(position))
        if sel_act is None:
            return
        elif sel_act is act_methods:
            self.yaml_methods = act_methods.isChecked()

        tree_widget.clear()
        menu.close()
        menu.deleteLater()

        self.refresh_object_tree()


    def object_tree_double_click(self):
        """
        Slot for double-click in the YAML/PKL tree view.
        """
        tree_widget = self.sender()

        item = tree_widget.currentItem()
        if not hasattr(item, 'tag'):
            return
        tag = item.tag
        if isinstance(tag, DenseDesignMatrix):
            self.mw.show_dataset(tag)
        elif isinstance(tag, Variable) or isinstance(tag, numpy.ndarray):
            self.mw.show_variable(tag)

    def refresh_object_tree(self):
        """
        Reloads the YAML tree.
        """        
        def divein_object(parent, oitr, depth):
            """
            Explore an object.
            """
            global interx
            try:
                interx = interx+ 1
                if interx == 254:
                    print interx

                # only expand the objects first time
                if isinstance(oitr, object):
                    try:
                        if oitr in self.flat_object_list:
                            return
                        self.flat_object_list.append(oitr)
                    except (TypeError, ValueError, AttributeError):
                        pass

                if isinstance(oitr, Tensor):
                    recurse_object(parent, dir(oitr), 
                                   self.yaml_max_depth-1, False, oitr)
                elif isinstance(oitr, Variable):
                    recurse_object(parent, dir(oitr),
                                   self.yaml_max_depth-1, False, oitr)
                elif isinstance(oitr, PYL2_TYPED):
                    recurse_object(parent, dir(oitr), depth+1, True, oitr)
                elif isinstance(oitr, DictType):
                    recurse_object(parent, oitr, depth+1, False, oitr)
                elif hasattr(oitr, '__iter__'):
                    recurse_object(parent, oitr, depth+1, False, oitr)
                elif isinstance(oitr, Model):
                    recurse_object(parent, dir(oitr), depth+1, True, oitr)
                else:
                    recurse_object(parent, dir(oitr), depth+1, True, oitr)
                    
            except Exception:
                logger.debug('Failed to dive in object', exc_info=True)

        def recurse_object(parent, obj, depth, force_dict=False, original_obj=None):
            """
            Explore the components of the object.
            """
            if depth >= self.yaml_max_depth:
                return
            try:
                index = -1
                for i in obj:
                    index = index + 1
                    b_recurse = True
                    tag = None

                    if isinstance(obj, DictType):
                        label = i
                        oitr = obj[i]
                    elif hasattr(obj, '__iter__') and not force_dict:
                        label = str(index)
                        oitr = i
                    else:
                        label = i
                        oitr = getattr(original_obj, i)

                    kind = type(oitr).__name__
                    value = ''

                    if label.startswith('__') and label.endswith('__'):
                        if self.yaml_filter_hidden:
                            continue
                    if isinstance(oitr, BuiltinFunctionType):
                        if not self.yaml_build_ins:
                            continue
                        b_recurse = False
                    if kind == 'method-wrapper' or kind == 'instancemethod':
                        if not self.yaml_methods:
                            continue
                        b_recurse = False
                        label = label + '()'
                    elif kind == 'ndarray':
                        b_recurse = False
                        value = str(oitr.shape)
                        tag = oitr

                    if isinstance(oitr, STRINGABLE_TYPES):
                        value = str(oitr)
                        b_recurse = False
                    elif isinstance(oitr, PYL2_TYPED):
                        value = str(oitr)
                        b_recurse = True
                        tag = oitr
                    elif kind == 'tuple':
                        value = str(oitr)
                        b_recurse = False
                    elif isinstance(oitr, Variable):
                        label = label + ' <' + str(oitr.name) + '>'
                        kind = kind + ' <' + str(oitr.type) + '>'
                        value = str(oitr.eval())
                        if len(value) > 20:
                            value = value[:20] + '...'
                        b_recurse = False
                        tag = oitr
                    if label == 'yaml_src':
                        if len(value) > 20:
                            value = value[:20] + '...'

                    #print '-'*depth, label, kind, value
                    tree_it = QtGui.QTreeWidgetItem(parent, 
                                                    [label, kind, value])
                    tree_it.tag = tag

                    if b_recurse:
                        divein_object(tree_it, oitr, depth)
            except (MissingInputError, RuntimeError, 
                    AttributeError, TypeError):
                pass

        self.flat_object_list = []
        self.tree_widget.clear()
        root = QtGui.QTreeWidgetItem(self.tree_widget,
                                     ['root', type(self.obj_tree).__name__])
        divein_object(root, self.obj_tree, 0)

        # expand first layer
        root.setExpanded(True)
        child_count = root.childCount()
        for i in range(child_count):
            item = root.child(i)
            item.setExpanded(True)
        self.tree_widget.resizeColumnToContents(0)
        self.tree_widget.resizeColumnToContents(1)
        