#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Nicu Tofan <nicu.tofan@gmail.com>
"""

from PyQt4 import QtGui, QtCore

from .gui import center

class RemoteDialog(QtGui.QDialog):
    """
    Allows selecting remote in order to debug on  that remote.

    """
    def __init__(self, mw):
        """
        Constructor
        """

        super(RemoteDialog, self).__init__()
        
        self.mw = mw

        self.init_ui()

    def init_ui(self):
        """
        Prepares the GUI.
        """
        self.resize(300, 200)
        self.setWindowTitle('Connect to remote')
        center(self)

        self.button_box = QtGui.QDialogButtonBox(self)
        self.button_box.setGeometry(QtCore.QRect(150, 250, 341, 32))
        self.button_box.setOrientation(QtCore.Qt.Horizontal)
        self.button_box.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.button_box.setObjectName("button_box")


        lbl_address = QtGui.QLabel('Address')
        lbl_post_rcv = QtGui.QLabel('Control port')
        lbl_port_sub = QtGui.QLabel('Broadcast port')

        le_address = QtGui.QLineEdit()
        le_address.setPlaceholderText('The address of the remote machine')
        le_address.setToolTip('This may also be an ip address.')
        le_address.setText('127.0.0.1')
        sp_port_rcv = QtGui.QSpinBox()
        sp_port_rcv.setMinimum(1024)
        sp_port_rcv.setMaximum(65565)
        sp_port_rcv.setValue(5955)
        sp_port_rcv.setToolTip('Port for command and control.')
        sp_port_sub = QtGui.QSpinBox()
        sp_port_sub.setMinimum(1024)
        sp_port_sub.setMaximum(65565)
        sp_port_sub.setValue(5956)
        sp_port_sub.setToolTip('Port where the remote debugger publishes information.')
        
        grid1 = QtGui.QGridLayout()
        grid1.setSpacing(10)
        
        grid1.addWidget(lbl_address, 1, 0)
        grid1.addWidget(le_address, 1, 1)
        grid1.addWidget(lbl_post_rcv, 2, 0)
        grid1.addWidget(sp_port_rcv, 2, 1)
        grid1.addWidget(lbl_port_sub, 3, 0)
        grid1.addWidget(sp_port_sub, 3, 1)

        grid = QtGui.QVBoxLayout()
        grid.setSpacing(10)

        grid.addLayout(grid1)
        grid.addWidget(self.button_box)
        self.setLayout(grid)

        QtCore.QObject.connect(self.button_box, QtCore.SIGNAL("accepted()"), self.accept)
        QtCore.QObject.connect(self.button_box, QtCore.SIGNAL("rejected()"), self.reject)   
        QtCore.QMetaObject.connectSlotsByName(self)
        
        self.le_address = le_address
        self.sp_port_rcv = sp_port_rcv
        self.sp_port_sub = sp_port_sub


    def get_values(self):
        """
        Return the values selected by the user.
        """
        values = {'address': self.le_address.text().strip(),
                  'rport': self.sp_port_rcv.value(),
                  'pport': self.sp_port_sub.value()}
        return values
