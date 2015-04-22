#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various support routines related to image manipulation.

@author: Nicu Tofan <nicu.tofan@gmail.com>
"""
import numpy
from PyQt4 import QtGui
from PIL.ImageQt import ImageQt
from scipy.misc.pilutil import toimage


def nparrayToQPixmap(array_image):
    """
    Converts an numpy image (W, H, 3) into a Qt pixmap.
    """

    pil_image = toimage(array_image)
    qtImage = ImageQt(pil_image)
    if len(array_image.shape) == 3:
        frm = QtGui.QImage.Format_ARGB32
    else:
        frm = QtGui.QImage.Format_Mono
    q_image = QtGui.QImage(qtImage).convertToFormat(frm)
    q_pixmap = QtGui.QPixmap(q_image)
    return q_pixmap

def gray2qimage(gray):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(gray.shape) != 2:
        raise ValueError("gray2QImage can only convert 2D arrays")

    gray = _normalize255(gray, normalize=255)
    gray = numpy.require(gray, numpy.uint8, 'C')

    height, width = gray.shape

    result = QtGui.QImage(gray.data, width, height, QtGui.QImage.Format_Indexed8)
    result.ndarray = gray
    for i in range(256):
        result.setColor(i, QtGui.QColor(i, i, i).rgb())
    q_pixmap = QtGui.QPixmap(result)
    return q_pixmap

def _normalize255(array, normalize=None, clip=None):
    """
    Brings the array values into requested range.

    Parameters
    ----------

    array : numpy.ndarray
        The array to convert.
    normalize : None, bool or (min,max), optional
        If it is True the array will be normalized between min and max.
        If it is an integer the array will be normalized between 0 and
        `normalize`.
        If it is a tuple, the array will be normalized between first and
        second component.
    clip : tuple, optional
        The clipping range.
    """
    if normalize:
        if normalize is True:
            normalize = array.min(), array.max()
        elif numpy.isscalar(normalize):
            normalize = (0, normalize)
        nmin, nmax = normalize

    if nmin:
        array = array - nmin

        if nmax != nmin:
            scale = 255. / (nmax - nmin)
        if scale != 1.0:
            array = array * scale

    if clip:
        low, high = clip
        numpy.clip(array, low, high, array)

    return array
