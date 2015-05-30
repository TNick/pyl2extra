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
import os
from PyQt4 import QtGui, QtCore

from pyl2extra.gui.guihelpers import center
from pyl2extra.gui.guihelpers import make_act
from pyl2extra.gui.imageqt import toimage as pil2qt
from pyl2extra.datasets.img_dataset.dataset import ImgDataset
from pyl2extra.datasets.img_dataset.data_providers import RandomProvider
from pyl2extra.datasets.img_dataset.data_providers import DictProvider
from pyl2extra.datasets.img_dataset.generators import InlineGen
from pyl2extra.datasets.img_dataset import adjusters
from pylearn2.utils.image import ndarray_from_pil

logger = logging.getLogger(__name__)
Q = QtGui.QMessageBox.question

class MainWindow(QtGui.QMainWindow):
    """
    Main application window.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.dataset = self.create_img_dataset()
        #: the list of list widgets
        self.qtimg = []

        self.init_actions()
        self.init_ui()

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
       
        self.lbl_info = QtGui.QLabel('Information')
        self.lblimg = QtGui.QLabel()
        self.lblimg.setMaximumSize(128, 128)
        grid_info = QtGui.QVBoxLayout()
        self.lbl_p = QtGui.QLabel('path: ')
        self.lbl_w = QtGui.QLabel('width: ')
        self.lbl_h = QtGui.QLabel('height: ')
        self.lbl_ch = QtGui.QLabel('channels: ')
        self.lbl_m = QtGui.QLabel('mode: ')
        grid_info.addWidget(self.lbl_p)
        grid_info.addWidget(self.lbl_w)
        grid_info.addWidget(self.lbl_h)
        grid_info.addWidget(self.lbl_ch)
        grid_info.addWidget(self.lbl_m)
        
        self.lst = self.create_img_list()
        
        grid.addWidget(self.lbl_info, 0, 0, 1, 1)
        grid.addWidget(self.lblimg, 0, 1, 1, 1)
        grid.addLayout(grid_info, 0, 2, 1, 1)
        grid.addWidget(self.lst, 1, 0, 1, 3)
        
        gridw.setLayout(grid)
        self.setCentralWidget(gridw)

    def create_img_list(self):
        """
        Create main image list used for presenting the results.
        """
        lst = QtGui.QListWidget()
        lst.setViewMode(QtGui.QListView.IconMode)
        lst.setUniformItemSizes(True)
        lst.setGridSize(QtCore.QSize(150, 150))
        lst.setIconSize(QtCore.QSize(128, 128))
        lst.setMovement(QtGui.QListView.Snap)

        return lst

    def create_img_dataset(self):
        """
        Create main image list used for presenting the results.
        """
        adjb = adjusters.BackgroundAdj(backgrounds=[(0, 0, 0),
                                                    (0, 0, 128),
                                                    (0, 128, 0),
                                                    (128, 0, 0),
                                                    (255, 255, 255),
                                                    (128, 128, 128)],
        #adjb = adjusters.BackgroundAdj(backgrounds=[(0, 0, 128)],
                                       image_files=None)
        adjq = adjusters.MakeSquareAdj(size=128, 
                                       order=3,
                                       construct=False,
                                       cache_loc=None)
        adjf = adjusters.FlipAdj(horizontal=True,
                                 vertical=True)
        adjr = adjusters.RotationAdj(min_deg=-45.0,
                                     max_deg=45.0,
                                     step=15.0,
                                     order=3,
                                     resize=False)
        adjs = adjusters.ScalePatchAdj(outsize=None,
                                       start_factor=0.8,
                                       end_factor=0.9,
                                       step=0.1,
                                       placements=None,
                                       order=3)
        adjg = adjusters.GcaAdj(start_scale=1.,
                                end_scale=1.,
                                step_scale=0.,
                                subtract_mean=None,
                                use_std=None,
                                start_sqrt_bias=0.,
                                end_sqrt_bias=0.,
                                step_sqrt_bias=0.)
        data_provider = RandomProvider(rng=None, 
                                       count=100, 
                                       alpha=False,
                                       size=(128, 128))
        generator = InlineGen(profile=False)
        dataset = ImgDataset(data_provider=data_provider,
                             adjusters=[adjq, adjs, adjr, adjf, adjb, adjg],
                             generator=generator,
                             shape=(128, 128),
                             axes=('b', 0, 1, 'c'), 
                             cache_loc=None,
                             rng=None)
        
        return dataset

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
        self.act_load_npy = make_act('Open &numpy ...', self,
                                     'ftp.png',
                                     'Ctrl+U',
                                     'Load a numpy array consisting of images',
                                     self.browse_npy)
        self.act_load_dir = make_act('Open &directory ...', self,
                                     'folders_explorer.png',
                                     'Ctrl+D',
                                     'Load directory and preprocess',
                                     self.browse_dir)
        menubar = self.menuBar()

        menu_file = menubar.addMenu('&File')
        menu_file.addAction(self.act_load_img)
        menu_file.addAction(self.act_load_npy)
        menu_file.addAction(self.act_load_dir)
        menu_file.addAction(self.act_exit)

        self.toolbar = self.addToolBar('General')
        self.toolbar.addAction(self.act_load_img)
        self.toolbar.addAction(self.act_load_npy)
        self.toolbar.addAction(self.act_load_dir)
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
            
    def flatten_descr(self, descr):
        """
        Converts a description returned by `ImgDataset.process_labels()` to
        a tooltip usable string.
        """
        result = []
        for ditem in descr:
            for key in ditem:
                result.append('%s: %s' % (key, ditem[key]))
        return result
        
    def load_npy(self, fname):
        """
        Slot that loads images from a numpy array.
        """
        if not fname:
            return

        try:
            self.lst.clear()
            
            self.lbl_p.setText(fname)
            batch = numpy.load(fname)
            
            if len(batch.shape) != 4:
                logger.error('Shape %s is unexpected for numpy array '
                             'in file %s',
                             str(batch.shape), fname)
                QtGui.QMessageBox.warning(self, 'Error',
                                          'Unexpected shape found inside '
                                          '%s file.' % (fname))
                return
            if not batch.shape[3] in (1, 2, 3, 4):
                logger.error('Expected layout is b01c with 4 channels '
                             'at most; file %s has %d channels.',
                             fname, batch.shape[3])
                QtGui.QMessageBox.warning(self, 'Error',
                                          'Unexpected shape found inside '
                                          '%s file (%d channels)' % 
                                          (fname, batch.shape[3]))
                return
            
            self.lbl_w.setText('width: %d' % batch.shape[2])
            self.lbl_h.setText('height: %d' % batch.shape[1])
            self.lbl_ch.setText('channels: %d' % batch.shape[3])
            self.lbl_m.setText('mode: numpy')
            self.lbl_info.setText(os.path.split(fname)[1])
            self.lblimg.setPixmap(QtGui.QPixmap())
            
            self._add_batch(batch)
            
        except Exception, exc:
            logger.error('Loading image file failed', exc_info=True)
            QtGui.QMessageBox.warning(self, 'Exception', str(exc))
        
    def load_dir(self, dname):
        """
        Slot that loads an image directory.
        """
        if not dname:
            return
        try:
            self.lst.clear()
            self.lbl_p.setText(dname)
            self.lbl_w.setText('width: -')
            self.lbl_h.setText('height: -')
            self.lbl_ch.setText('channels: -')
            self.lbl_m.setText('mode: ')
            self.lbl_info.setText(dname)
            self.lblimg.setPixmap(QtGui.QPixmap())
            
            data = {}
            allowed_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff')
            for fname in os.listdir(dname):
                fn, ext = os.path.splitext(fname)
                if len(ext) == 0:
                    continue
                ext = ext[1:].lower()
                if not ext in allowed_extensions:
                    continue
                fname = os.path.join(dname, fname)
                data[fname] = ''
            provider = DictProvider(data=data)
            
            for fname in data:
                
                fname = os.path.join(dname, fname)
                batch = provider.read_image(fname)
                batch = batch.reshape(1, batch.shape[0], 
                                      batch.shape[1],
                                      batch.shape[2])
                batch = self.dataset.process(batch, accumulate=False)
                ibatch = batch[:, :, :, 0:3]
                ibatch = numpy.cast['uint8'](255.0 * (ibatch - ibatch.min()) / 
                                             (ibatch.max() - ibatch.min()))
                ibatch = ibatch.reshape(batch.shape[1], 
                                        batch.shape[2],
                                        batch.shape[3])
                self.img_item(ibatch)
            
        except Exception, exc:
            logger.error('Loading image file failed', exc_info=True)
            QtGui.QMessageBox.warning(self, 'Exception', str(exc))
        
    def load_img(self, fname):
        """
        Slot that loads an image file.
        """
        if not fname:
            return
        try:
            self.lst.clear()
            
            img = Image.open(fname)
            self.lbl_p.setText(fname)
            self.lbl_w.setText('width: %d' % img.size[0])
            self.lbl_h.setText('height: %d' % img.size[1])
            self.lbl_ch.setText('channels: %d' % len(img.getbands()))
            self.lbl_m.setText('mode: %s' % img.mode)
            self.lbl_info.setText(os.path.split(fname)[1])
            
            qtimg = pil2qt(img)
            qtpix = QtGui.QPixmap(qtimg.image)
            w = min(qtpix.width(), self.lblimg.maximumWidth())
            h = min(qtpix.height(), self.lblimg.maximumHeight())
            qtpix = qtpix.scaled(w, h, 
                                 QtCore.Qt.KeepAspectRatio,
                                 QtCore.Qt.SmoothTransformation)
            self.lblimg.setPixmap(qtpix)
            
            batch = self.dataset.data_provider.read_image(img)
            batchin = batch.reshape(1, batch.shape[0], batch.shape[1], batch.shape[2])
            batch = self.dataset.process(batchin, accumulate=True)
            
            description = self.dataset.process_labels(1)
            self._add_batch(batch, description)
        except Exception, exc:
            logger.error('Loading image file failed', exc_info=True)
            QtGui.QMessageBox.warning(self, 'Exception', str(exc))

    def _add_batch(self, batch, description=None):
        """
        Expects a batch in b01c format. Creates images for items in batch.
        """
        for i in range(batch.shape[0]):
            ibatch = batch[i, :, :, 0:3]
            ibatch = numpy.cast['uint8'](255.0 * (ibatch - ibatch.min()) / 
                                         (ibatch.max() - ibatch.min()))      
            tooltip = None
            if not description is None:
                tooltip = self.flatten_descr(description[i])
            self.img_item(ibatch, tooltip=tooltip)        


    def img_item(self, img, label=None, tooltip=None):
        """
        Create an list item from an Image.
        """
        lsit = QtGui.QListWidgetItem()
        lsit.setSizeHint(QtCore.QSize(148, 148))
        if not label is None:
            lsit.setText(label)
        lsit.setBackgroundColor(QtGui.QColor('grey'))
        if isinstance(img, Image.Image):
            qtimg = pil2qt(img)
        elif isinstance(img, numpy.ndarray):
            mnm = img.min()
            mx = img.max()
            if len(img.shape) == 1:
                raise NotImplementedError()
            elif len(img.shape) == 2:
                raise NotImplementedError()
            elif len(img.shape) == 3:
                if img.dtype == 'uint8':
                    pass
                elif mnm >= -5 and mx < 260 and mx > 10:
                    img = img * (img > 0)
                    lg = (img > 255)
                    img = img * lg * (-1) + img + (lg * 255)
                    img = numpy.cast['uint8'](img)
                elif img.shape[2] == 1:
                    img = ((img - mnm) / (mx - mnm)) * 255
                    img = numpy.cast['uint8'](img)
                elif img.shape[2] == 3:
                    img = ((img - mnm) / (mx - mnm)) * 255
                    img = numpy.cast['uint8'](img)
                else:
                    img = img[:, :, 0:3]
                    mnm = img.min()
                    mx = img.max()
                    img = ((img - mnm) / (mx - mnm)) * 255
                    img = numpy.cast['uint8'](img)
                    
                    #raise NotImplementedError()
            else:
                raise ValueError('Cannot display array of shape %s' % 
                                 str(img.shape))
            # print mnm, mx, img.shape, img.dtype
            img = Image.fromarray(img)
            qtimg = pil2qt(img)
        else:
            raise ValueError('%s not dupported by img_item()' %
                             str(img.__class__))
        qtpix = QtGui.QPixmap(qtimg.image)
        qticn = QtGui.QIcon(qtpix)
        lsit.setIcon(qticn)
        if tooltip:
            lsit.setToolTip('\n'.join(tooltip))
            lsit.setWhatsThis(', '.join(tooltip))
        self.lst.addItem(lsit)
        self.qtimg.append(qtimg)
        return lsit
        
    def browse_img(self):
        """
        Slot that browse for and loads an image file.
        """
        fname = QtGui.QFileDialog.getOpenFileName(self,
                                                  'Open image file')
        if not fname:
            return
        self.load_img(fname)
        
    def browse_dir(self):
        """
        Slot that browse for and loads an image directory.
        """
        fname = QtGui.QFileDialog.getExistingDirectory(self,
                                                  'Open image file')
        if not fname:
            return
        self.load_dir(fname)
        
    def browse_npy(self):
        """
        Slot that browse for and loads an numpy file.
        """
        fname = QtGui.QFileDialog.getOpenFileName(self,
                                                  'Open numpy (.npy) file')
        if not fname:
            return
        self.load_npy(fname)
