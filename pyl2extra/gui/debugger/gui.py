#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Nicu Tofan <nicu.tofan@gmail.com>
"""

from PyQt4 import QtGui

def center(widget):
    """
    Place a widget at the center of the screen.
    """
    frame_geom = widget.frameGeometry()
    center_point = QtGui.QDesktopWidget().availableGeometry().center()
    frame_geom.moveCenter(center_point)
    widget.move(frame_geom.topLeft())
