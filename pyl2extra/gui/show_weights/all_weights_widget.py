#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Widget that can show all weights on a single image.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from copy import copy
import logging
import numpy
import os
import Image
from PyQt4 import QtGui, QtCore
from pyqtgraph import ColorButton
import sys

from pyl2extra.gui.image_viewer import (ActionsMixin, SettingsMixin,
                                        NavigMixin, ImageViewer)
from pyl2extra.gui.guihelpers import get_icon

logger = logging.getLogger(__name__)
Q = QtGui.QMessageBox.question


class AllWeightsWidget(QtGui.QMainWindow, ActionsMixin,
                       SettingsMixin, NavigMixin):
    """
    Widget capable of generating a weights report

    Parameters
    ----------
    value : numpy.ndarray
        The array to explore.
    auto_refresh : bool
        Wether to autorefresh on each change or not.
    """

    def __init__(self, value=None, auto_refresh=None):
        #super(MainWindow, self).__init__()
        QtGui.QMainWindow.__init__(self)
        ActionsMixin.__init__(self)
        SettingsMixin.__init__(self)
        NavigMixin.__init__(self)

        if auto_refresh is None:
            auto_refresh = self.settings.value('auto_refresh', False) == 'true'

        #: used to block signals if the user does not want auto-refresh
        self.signal_lock = not auto_refresh
        #: holds the last numpy image generated
        self.crt_image = None
        #: the array we're presenting (may be None)
        self.value = None

        self.init_ui()

        self.signal_lock = True
        self.set_value(value)
        self.read_transf_settings()
        self.signal_lock = not auto_refresh
        self._internal_refresh()

    def init_ui(self):
        """
        Prepare GUI for main window.
        """
        gridw = QtGui.QWidget()
        grid = QtGui.QGridLayout()
        grid.setSpacing(15)

        info = 'Order of the axes (0 based) '
        self.lbl_order = QtGui.QLabel('Order')
        self.lbl_order.setWhatsThis(info)
        self.lbl_order.setToolTip(info)
        grid.addWidget(self.lbl_order, 0, 0, 1, 1)
        self.le_order = QtGui.QLineEdit()
        self.le_order.setWhatsThis(info)
        self.le_order.setToolTip(info)
        self.le_order.setMaximumWidth(150)
        self.connect(self.le_order, QtCore.SIGNAL('editingFinished()'),
                     self.regenerate)
        grid.addWidget(self.le_order, 0, 1, 1, 1)

        info = 'Spacing between images in pixels'
        self.lbl_spacing = QtGui.QLabel('Spacing')
        self.lbl_spacing.setWhatsThis(info)
        self.lbl_spacing.setToolTip(info)
        grid.addWidget(self.lbl_spacing, 1, 0, 1, 1)
        self.sp_spacing = QtGui.QSpinBox()
        self.sp_spacing.setWhatsThis(info)
        self.sp_spacing.setToolTip(info)
        self.sp_spacing.setMinimum(0)
        self.sp_spacing.setMaximum(100)
        self.sp_spacing.setValue(1)
        self.sp_spacing.setMaximumWidth(150)
        self.connect(self.sp_spacing, QtCore.SIGNAL('valueChanged(int)'),
                     self.regenerate)
        grid.addWidget(self.sp_spacing, 1, 1, 1, 1)

        info = 'Size of one value point in pixels'
        self.lbl_psz = QtGui.QLabel('Point size')
        self.lbl_psz.setWhatsThis(info)
        self.lbl_psz.setToolTip(info)
        grid.addWidget(self.lbl_psz, 2, 0, 1, 1)
        self.sp_psz = QtGui.QSpinBox()
        self.sp_psz.setWhatsThis(info)
        self.sp_psz.setToolTip(info)
        self.sp_psz.setMinimum(1)
        self.sp_psz.setMaximum(10000)
        self.sp_psz.setValue(2)
        self.sp_psz.setMaximumWidth(150)
        self.connect(self.sp_psz, QtCore.SIGNAL('valueChanged(int)'),
                     self.regenerate)
        grid.addWidget(self.sp_psz, 2, 1, 1, 1)

        info = 'Base color for the background (bottom-most layer)'
        self.lbl_bk = QtGui.QLabel('Background')
        self.lbl_bk.setWhatsThis(info)
        self.lbl_bk.setToolTip(info)
        grid.addWidget(self.lbl_bk, 3, 0, 1, 1)
        self.col_bk = ColorButton(color=(150, 150, 200))
        self.col_bk.setWhatsThis(info)
        self.col_bk.setToolTip(info)
        self.connect(self.col_bk, QtCore.SIGNAL('sigColorChanged()'),
                     self.regenerate)
        grid.addWidget(self.col_bk, 3, 1, 1, 1)

        info = 'Shape of the original input'
        if self.value is None:
            orig_shp = '-'
        else:
            orig_shp = ', '.join([str(i) for i in self.value.shape])
        self.lbl_orig = QtGui.QLabel('Original shape: %s' % orig_shp)
        self.lbl_orig.setWhatsThis(info)
        self.lbl_orig.setToolTip(info)
        grid.addWidget(self.lbl_orig, 4, 0, 2, 1)

        info = 'Shape of transformed value according to *order*'
        self.lbl_transf = QtGui.QLabel('Transformed: -')
        self.lbl_transf.setWhatsThis(info)
        self.lbl_transf.setToolTip(info)
        grid.addWidget(self.lbl_transf, 5, 0, 2, 1)

        spat = QtGui.QSpacerItem(10, 100,
                                 QtGui.QSizePolicy.Minimum,
                                 QtGui.QSizePolicy.Expanding)
        grid.addItem(spat, 6, 0, 2, 1)

        self._imageViewer = ImageViewer()
        self._imageViewer.enableHandDrag(True)
        grid.addWidget(self._imageViewer, 0, 3, 20, 8)

        gridw.setLayout(grid)
        self.setCentralWidget(gridw)

        self.createActions()
        self.createMenus()
        self.createToolbars()

        self.autorefresh_act = QtGui.QAction(
            get_icon('ecommerce_server.png'),
            "Auto-refresh", self,
            shortcut="Ctrl+F5",
            checkable=True,
            triggered=self._autorefresh_change)
        self.autorefresh_act.setChecked(not self.signal_lock)

        self.refresh_act = QtGui.QAction(
            get_icon('eye.png'),
            "Refresh", self,
            shortcut="F5",
            triggered=self._internal_refresh)

        self.tb_refresh = self.addToolBar('Refresh')
        self.tb_refresh.addAction(self.autorefresh_act)
        self.tb_refresh.addAction(self.refresh_act)

        self.save_act = QtGui.QAction(
            get_icon('download.png'),
            "Save As...", self,
            shortcut="Ctrl+S",
            triggered=self.save_as)
        self.fileMenu.insertAction(self.exitAct, self.save_act)

    def save_as(self):
        """Save current image as a file."""
        if self.crt_image is None:
            QtGui.QMessageBox.warning(self, "Error",
                                      "No image to save")
            return
        path = os.path.join(self.settings.value('aww/save_path', ''),
                            'Untitled.png')
        fname = QtGui.QFileDialog.getSaveFileName(parent=self,
                                                  caption='Select location',
                                                  directory=path,
                                                  filter='*.*')
        if fname is None or len(fname) == 0:
            return
        path, name = os.path.split(fname)
        self.settings.setValue('aww/save_path', path)
        try:
            img = Image.fromarray(self.crt_image)
            img.save(fname)
        except Exception, exc:
            QtGui.QMessageBox.warning(self, "Error",
                                      "Failed to save image\n%s" % exc.message)


    def _autorefresh_change(self):
        """Value of the autorefresh has changed."""
        self.signal_lock = not self.autorefresh_act.isChecked()

    def read_transf_settings(self):
        """Find out if there are settings for this value and load them."""
        if self.value is None:
            return

        var_key = 'lys/shape_' + '_'.join([str(i) for i in self.value.shape])
        self.settings.beginGroup(var_key)
        if self.settings.contains('background'):
            try:
                self.le_order.setText(self.settings.value('order'))
                self.sp_spacing.setValue(int(self.settings.value('spacing')))
                self.sp_psz.setValue(int(self.settings.value('ptsz')))
                col = self.settings.value('background').split(',')
                col = [int(i) for i in col]
                self.col_bk.setColor(col)
            except (ValueError, KeyError):
                pass
        self.settings.endGroup()

    def save_transf_settings(self):
        """Save the settings based on shape of the value."""
        if self.value is None:
            return
        var_key = 'lys/shape_' + '_'.join([str(i) for i in self.value.shape])
        self.settings.beginGroup(var_key)

        self.settings.setValue('order', self.le_order.text())
        self.settings.setValue('spacing', self.sp_spacing.value())
        self.settings.setValue('ptsz', self.sp_psz.value())
        collst = ','.join([str(i) for i in self.col_bk.color(mode='byte')])
        self.settings.setValue('background', collst)

        self.settings.endGroup()

    def _internal_refresh(self):
        """
        Explicitly requested refresh.
        """
        prev_lock = self.signal_lock
        self.signal_lock = False
        self.regenerate()
        self.signal_lock = prev_lock

    def set_value(self, value):
        """
        Change the value that we're showing.
        """
        self.value = value

        self.lbl_transf.setText('Transformed: -')
        if not value is None:
            ordstr = ', '.join([str(i) for i in range(len(self.value.shape))])
            self.le_order.setText(ordstr)
            self.le_order.setEnabled(True)
            self.sp_spacing.setEnabled(True)
            orig_shp = ', '.join([str(i) for i in self.value.shape])
            self.lbl_orig.setText('Original shape: %s' % orig_shp)
            self.regenerate()
        else:
            self.le_order.setText('')
            self.le_order.setEnabled(False)
            self.sp_spacing.setEnabled(False)
            self.lbl_orig.setText('Original shape: -')

    def validate_order(self):
        """
        Get the order.
        """
        try:
            order = [int(i) for i in self.le_order.text().split(',')]
        except ValueError:
            QtGui.QMessageBox.warning(self, "Error",
                                      "Can't convert %s into a list or integer"
                                      % str(self.le_order.text()))
            return None

        for i in order:
            if i < 0 or i >= len(self.value.shape):
                QtGui.QMessageBox.warning(self, "Error",
                                          "%d is outside vali8d range (0-%d)"
                                           % (i, len(self.value.shape)-1))
                return None
        for i in range(len(self.value.shape)):
            if not i in order:
                order.append(i)
        if len(order) != len(self.value.shape):
            QtGui.QMessageBox.warning(self, "Error",
                                      "Duplicate entries in %s"
                                      % self.le_order.text())
            return None
        return order

    def regenerate(self):
        """
        Regenerate the image.
        """
        if self.signal_lock:
            return
        background = self.col_bk.color(mode='byte')
        separator = self.sp_spacing.value()
        point_size = self.sp_psz.value()
        order = self.validate_order()
        if order is None:
            return
        img, reordered_shape = generate(separator, point_size, order,
                                        self.value, background)
        self.crt_image = img
        pixmap, self._keep_img = ImageViewer.pixmap_from_array(img)
        self._imageViewer.pixmap = pixmap
        self._imageViewer.fitHeight()

        rshp = ', '.join([str(i) for i in reordered_shape])
        self.lbl_transf.setText('Transformed: %s' % rshp)

    def closeEvent(self, event):
        """Overrides close event to save application settings."""
        self.save_transf_settings()
        self.settings.setValue('auto_refresh',
                               self.autorefresh_act.isChecked())
        event.accept()


def reorder(value, order):
    """
    Get the order.
    """
    value = value.copy()
    cutoff = len(order) - 1
    for i in range(cutoff):
        oit = order[i]
        if oit != i:
            value = value.swapaxes(i, oit)
            for j in range(i+1, len(order)):
                if order[j] == i:
                    order[j] = oit
                    break
    return value

def best_square_shape(number):
    """
    Compute best square shape or the very first if none are square.
    """
    number = int(number)
    if number < 0:
        sign = -1
    else:
        sign = 1
    number = number * sign
    candidates = []
    for i in range(1, number):
        if number % i == 0:
            cd2 = number / i
            cd2 = (i, cd2) if i < cd2 else (cd2, i)
            candidates.append(cd2)
    best = None
    for cand in candidates:
        if best is None:
            best = cand
        else:
            if (best[1] - best[0]) > (cand[1] - cand[0]):
                best = cand
    return (best[0] * sign, best[1])

ALLOW_COLORS = True
def best_shape(number, color_candidate=False):
    """
    Compute best arrangement.
    
    Parameters
    ----------
    number : int
        The number of items to arrange
    color_candidate : bool
        Should this be considered for colorized special case?
        
    Returns
    -------
    first : int
        The number of rows.
    second : int
        The number of columns.
        
    Notes
    -----
    If ``color_candidate`` is ``Treu`` and 
    ``ALLOW_COLORS`` is ``True``  and ``number`` is 3 a special value
    is returned - ``(None, 1)`` that means the implementation should 
    onsider the three elements Red, Green and Blue.
    """
    if color_candidate and ALLOW_COLORS and number == 3:
        return (None, 1)

    number = int(number)
    if number < 0:
        sign = -1
    else:
        sign = 1
    number = number * sign
    best_score = number * number
    best = [-1, -1]
    limit = int(number/2) + 2
    for i in range(1, limit):
        for j in range(i, limit):
            diff = i * j - number
            if diff < 0:
                continue
            score = (j - i) + diff
            if score < best_score:
                best_score = score
                best[0] = i
                best[1] = j
    return (best[0] * sign, best[1])

# how is the background to be updated on next level
UPDATE_RED = 10
UPDATE_GREEN = 10
UPDATE_BLUE = 10
UPDATE_ALPHA = 0

def generate(separator, point_size, order, value, back_value=None):
    """
    Regenerate the image.
    """
    if back_value is None:
        back_value = [200, 200, 50, 255]
    else:
        back_value = list(back_value)

    value = reorder(value, order)
    reordered_shape = value.shape
    value = 255 * (value - value.min()) / (value.max() - value.min())
    value = numpy.cast['uint8'](value)

    # the list of axes that we have to arrange is the shape without
    # last two axes that are shown to the user as individual images
    shapes = value.shape[:-2]

    # compute arrangements for these axes
    layouts = [best_shape(shp, i == len(shapes)-1) for i, shp in enumerate(shapes)]

    # compute the size of each rectangle for each axis
    lay_rev = [itm for itm in reversed(layouts)]
    width_prev = value.shape[-1] * point_size
    height_prev = value.shape[-2] * point_size
    dims = [(width_prev, height_prev, separator)]
    for i in range(len(shapes)):
        cols, rows = lay_rev[i]
        if cols != None:
            width_prev = cols * width_prev + (cols + 1) * separator
            height_prev = rows * height_prev + (rows + 1) * separator
        dims.append((width_prev, height_prev, separator))
    dims = [itm for itm in reversed(dims)]

    # the size of the image is width_prev x height_prev
    # create an image large enough and init to background
    img = numpy.empty(shape=(height_prev, width_prev, 4),
                      dtype='uint8')

    # color used for the "background"
    img[:, :, 0] = back_value[0]
    img[:, :, 1] = back_value[1]
    img[:, :, 2] = back_value[2]
    img[:, :, 3] = back_value[3]

    def paint_cmp(comp, bkg_color, level=0, deltax=0, deltay=0):
        """Recursive dig into axes"""
        if len(comp.shape) == 2:
            # we're actually drawing the image
            width, height, separator = dims[level]
            psx = deltax
            psy_orig = deltay

            for pszx in range(point_size):
                psy = psy_orig
                for pszy in range(point_size):
                    psx2 = psx + width
                    psy2 = psy + height
                    img[psy:psy2:point_size, psx:psx2:point_size, 0] = comp
                    img[psy:psy2:point_size, psx:psx2:point_size, 1] = comp
                    img[psy:psy2:point_size, psx:psx2:point_size, 2] = comp
                    img[psy:psy2:point_size, psx:psx2:point_size, 3] = 255
                    psy = psy + 1
                psx = psx + 1
        else:
            # update the background for next level
            bkg_color[0] = bkg_color[0] + UPDATE_RED
            bkg_color[1] = bkg_color[1] + UPDATE_GREEN
            bkg_color[2] = bkg_color[2] + UPDATE_BLUE
            bkg_color[3] = bkg_color[3] + UPDATE_ALPHA

            cols, rows = layouts[level]
            if cols is None:
                if rows == 1:
                    level = level + 1
                    width, height, separator = dims[level]
                    
                    psx = deltax
                    psy_orig = deltay
        
                    for pszx in range(point_size):
                        psy = psy_orig
                        for pszy in range(point_size):
                            psx2 = psx + width
                            psy2 = psy + height
                            img[psy:psy2:point_size, psx:psx2:point_size, 0] = comp[0]
                            img[psy:psy2:point_size, psx:psx2:point_size, 1] = comp[1]
                            img[psy:psy2:point_size, psx:psx2:point_size, 2] = comp[2]
                            img[psy:psy2:point_size, psx:psx2:point_size, 3] = 255
                            psy = psy + 1
                        psx = psx + 1
                    
                else:
                    raise RuntimeError('Unknown special case %d' % rows)
                return
                
            width, height, separator = dims[level+1]
            row = 0
            col = 0
            i = 0
            limit = comp.shape[0]
            for row in range(rows):
                for col in range(cols):
                    # compute the psition for this rectangle
                    psx = deltax + col * width + (col + 1) * separator
                    psy = deltay + row * height + (row + 1) * separator
                    psx2 = psx + width
                    psy2 = psy + height
                    img[psy:psy2, psx:psx2, 0] = bkg_color[0]
                    img[psy:psy2, psx:psx2, 1] = bkg_color[1]
                    img[psy:psy2, psx:psx2, 2] = bkg_color[2]
                    img[psy:psy2, psx:psx2, 3] = bkg_color[3]

                    paint_cmp(comp[i],
                              copy(back_value),
                              level=level+1,
                              deltax=psx,
                              deltay=psy)
                    i = i + 1
                    if i >= limit:
                        break
    paint_cmp(value, back_value)
    return img, reordered_shape

if __name__ == '__main__':
    var_test = numpy.random.rand(7, 4, 3, 5)
    app = QtGui.QApplication(sys.argv)
    ex = AllWeightsWidget(var_test)
    ex.show()
    sys.exit(app.exec_())
