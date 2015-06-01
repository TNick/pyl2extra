#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Nicu Tofan <nicu.tofan@gmail.com>
"""

from os import path
from PyQt4 import QtGui

def center(widget):
    """
    Place a widget at the center of the screen.

    Parameters
    ----------
    widget : PyQt4.QtGui.QWidget
        The panel to center on the screen.
    """
    frame_geom = widget.frameGeometry()
    center_point = QtGui.QDesktopWidget().availableGeometry().center()
    frame_geom.moveCenter(center_point)
    widget.move(frame_geom.topLeft())

def get_icon(icon_name):
    """
    Create a QIcon from an image in local resource directory

    Parameters
    ----------
    icon_name : str
        Name of the image including the extension.
    """
    icon_path = path.join(path.dirname(path.abspath(__file__)),
                          'resources',
                          icon_name)
    icon = QtGui.QIcon(icon_path)
    return icon

def make_act(label, owner, icon_name=None, shortcut=None, tip=None, slot=None):
    """
    Creates a QAction.

    Parameters
    ----------
    label : str
        The text to show with this action.
    owner : PyQt4.QtCore.QObject
        Owner of this instance.
    icon_name : str, optional
        The name of the icon to show (from resources) or None for no icon.
    shortcut : str, optional
        Qt-style shortcut.
    tip : str, optional
        Tooltip for the action.
    slot : str
        The name of a slot for ``triggered()`` event.
    """
    if icon_name:
        icon = get_icon(icon_name)
    else:
        icon = None

    act_result = QtGui.QAction(icon, label, owner)

    if shortcut:
        act_result.setShortcut(shortcut)
    if tip:
        act_result.setStatusTip(tip)
    if slot:
        act_result.triggered.connect(slot)

    return act_result
