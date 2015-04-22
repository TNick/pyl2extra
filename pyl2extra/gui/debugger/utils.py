#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Nicu Tofan <nicu.tofan@gmail.com>
"""

from os import path
from PyQt4 import QtGui

def get_icon(icon_name):
    icon_path = path.join(path.dirname(path.abspath(__file__)),
                          'resources',
                          icon_name)
    icon = QtGui.QIcon(icon_path)
    return icon

def make_act(label, owner, icon_name=None, shortcut=None, tip=None, slot=None):
    """
    Creates a QAction.
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

