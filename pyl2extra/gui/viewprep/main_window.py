#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main window for preprocessing app.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"


import Image
import logging
import numpy
from PyQt4 import QtGui, QtCore

from pyl2extra.gui.guihelpers import center
from pyl2extra.gui.guihelpers import make_act
from pyl2extra.gui.guihelpers import get_icon
from pyl2extra.gui.imageqt import toimage as pil2qt

logger = logging.getLogger(__name__)
Q = QtGui.QMessageBox.question

class MainWindow(QtGui.QMainWindow):
    """
    Main application window.
    """
    def __init__(self):
        """
        Constructor.
        """
        super(MainWindow, self).__init__()
        
        #: the list of list widgets
        self.lists = []
        self.qtimg = []

        self.init_actions()
        self.init_ui()
        self.load_img("/media/tnick/Big_mamma1/prog/spotally/samples/blazers/BZ111.jpg")

    def init_ui(self):
        """
        Prepare GUI for main window.
        """
        self.resize(900, 800)
        center(self)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle('Vizualize preprocessing')


        gridw = QtGui.QWidget()
        grid = QtGui.QGridLayout()
        grid.setSpacing(5)

        self.create_img_list(grid, 1, 'GCN')
        self.create_img_list(grid, 2, 'ZCN')
        self.create_img_list(grid, 3, 'Flip')
        self.create_img_list(grid, 4, 'Rotate')
        self.create_img_list(grid, 5, 'Scale')
        

        gridw.setLayout(grid)
        self.setCentralWidget(gridw)

    def init_actions(self):
        """
        Prepares the actions, then menus & toolbars.
        """

        self.act_exit = make_act('&Exit', self,
                                 'door_in.png',
                                 'Ctrl+Q',
                                 'Exit application',
                                 self.close)
        self.act_load_img = make_act('Open &image ...', self,
                                     'folder_go.png',
                                     'Ctrl+O',
                                     'Load file and preprocess',
                                     self.browse_img)

        menubar = self.menuBar()

        menu_file = menubar.addMenu('&File')
        menu_file.addAction(self.act_load_img)
        menu_file.addAction(self.act_exit)

        self.toolbar = self.addToolBar('General')
        self.toolbar.addAction(self.act_load_img)
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
            event.accept()
            QtCore.QCoreApplication.exit(0)
        else:
            event.ignore()

    def create_img_list(self, grid, index, label=''):
        """
        
        """
        lst = QtGui.QListWidget()
        lst.setViewMode(QtGui.QListView.IconMode)
        lst.setUniformItemSizes(True)
        lst.setGridSize(QtCore.QSize(150, 150))
        lst.setIconSize(QtCore.QSize(128, 128))
        lst.setMovement(QtGui.QListView.Snap)
        lbl = QtGui.QLabel(label)
        
        grid.addWidget(lbl, index, 0, 1, 1)
        grid.addWidget(lst, index, 1, 1, 1)
        self.lists.append(lst)
        return lst

    def img_item(self, label, img, lst):
        """
        """
        lsit = QtGui.QListWidgetItem()
        lsit.setSizeHint(QtCore.QSize(148, 148))
        lsit.setText(label)
        lsit.setBackgroundColor(QtGui.QColor('grey'))
        qtimg = pil2qt(img)
        qtpix = QtGui.QPixmap(qtimg.image)
        qticn = QtGui.QIcon(qtpix)
        lsit.setIcon(qticn)
        lst.addItem(lsit)
        self.qtimg.append(qtimg)

    def load_img(self, fname):
        """
        Slot that loads an image file.
        """
        if not fname:
            return
        try:
            img = Image.open(fname)
            
            from pylearn2.expr.preprocessing import global_contrast_normalize
            from pylearn2.utils.image import pil_from_ndarray
            from pylearn2.utils.image import ndarray_from_pil
            
            #imarray = numpy.array(img)
            imarray = ndarray_from_pil(img)
            if len(imarray.shape) == 3:
                imresh = imarray.reshape(imarray.shape[0]*imarray.shape[1], 
                                         imarray.shape[2]).swapaxes(0, 1)
                imresh = global_contrast_normalize(imresh, subtract_mean=True)
                def make_rgb(imgarry):
                    for i in range(imgarry.shape[0]):
                        arr_min = imgarry[i].min()
                        arr_max = imgarry[i].max()
                        imgarry[i] = (imgarry[i] - arr_min) * \
                                     255 / (arr_max - arr_min)
                
                make_rgb(imresh)
                imresh = imresh.swapaxes(0, 1)
                imresh = imresh.reshape(imarray.shape[0],
                                        imarray.shape[1],
                                        imarray.shape[2])
                imresh = imresh.astype('uint8')
            else:
                assert False            
            
            img.thumbnail((128,128), Image.ANTIALIAS)
            self.img_item("Original", img, self.lists[0])
            img = pil_from_ndarray(imresh)
            self.img_item("GCN", img, self.lists[0])
            
            img_cmp = imarray[:,:,0].swapaxes(0, 1)
            #img_cmp = img_cmp.reshape(imarray.shape[0]*imarray.shape[1], 1)
            img_cmp = numpy.array([img_cmp, img_cmp, img_cmp]).swapaxes(0, 2)
            img = Image.fromarray(img_cmp)
            self.img_item("Red", img, self.lists[0])
            
            img_cmp = imarray[:,:,1].swapaxes(0, 1)
            #img_cmp = img_cmp.reshape(imarray.shape[0]*imarray.shape[1], 1)
            img_cmp = numpy.array([img_cmp, img_cmp, img_cmp]).swapaxes(0, 2)
            img = Image.fromarray(img_cmp)
            self.img_item("Blue", img, self.lists[0])
            
            img_cmp = imarray[:,:,2].swapaxes(0, 1)
            #img_cmp = img_cmp.reshape(imarray.shape[0]*imarray.shape[1], 1)
            img_cmp = numpy.array([img_cmp, img_cmp, img_cmp]).swapaxes(0, 2)
            img = Image.fromarray(img_cmp)
            self.img_item("Green", img, self.lists[0])
            
            
            from pylearn2.datasets.preprocessing import ZCA
            from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
            
            imgzca = imarray.reshape(1,
                                     imarray.shape[0] *
                                     imarray.shape[1] *
                                     imarray.shape[2])            
            imgzca = (imgzca.T - imgzca.mean(axis=1)).T
            imgzca = (imgzca.T / numpy.sqrt(numpy.square(imgzca).sum(axis=1))).T
            imgzca *= float(55)
            

            ddm = DenseDesignMatrix(X=imgzca)
            zca = ZCA()
            zca.apply(ddm, can_fit=True)
            imgzca = ddm.get_design_matrix()
            imgzca = imarray.reshape(imarray.shape[0],
                                     imarray.shape[1],
                                     imarray.shape[2])
            imgzca.swapaxez(0, 1)
            make_rgb(imgzca)
            imgzca.swapaxez(0, 1)
            img = Image.fromarray(imgzca)
            self.img_item("ZCA", img, self.lists[1])
            
            
#            lsit = QtGui.QListWidgetItem()
#            lsit.setSizeHint(QtCore.QSize(128, 128))
#            lsit.setText("0")
#            lsit.setBackgroundColor(QtGui.QColor('grey'))
#            qtimg = pil2qt(img)
#            qtpix = QtGui.QPixmap(qtimg.image)
#            qticn = QtGui.QIcon(qtpix)
#            lsit.setIcon(qticn)
#            self.lists[0].addItem(lsit)
            
#            
#            lsit = QtGui.QListWidgetItem()
#            lsit.setText("1")
#            lsit.setBackgroundColor(QtGui.QColor('grey'))
#            lsit.setIcon(get_icon('ftp.png'))
#            self.lists[0].addItem(lsit)
#            
        except Exception, exc:
            logger.error('Loading image file failed', exc_info=True)
            QtGui.QMessageBox.warning(self, 'Exception', str(exc))

    def browse_img(self):
        """
        Slot that browse for and loads an image file.
        """
        fname = QtGui.QFileDialog.getOpenFileName(self,
                                                  'Open image file')
        if not fname:
            return
        self.load_img(fname)
