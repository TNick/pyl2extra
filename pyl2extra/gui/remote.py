#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A dialog that allows the user to choose connection settings.

It provides a GUI to :class:`pyl2extra.utils.remote.Remote`.
"""
from __future__ import with_statement

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"


from PyQt4 import (QtCore, QtGui)
import sys

from pyl2extra.utils import remote


class Remote(QtGui.QDialog):
    """
    Dialog to allow editing remote machine connection settings.

    Parameters
    ----------
    parent : str, optional
        Parent widget.
    remt : str, optional
        The remote in question.
    """
    def __init__(self, parent=None, remt=None):

        if remt is None:
            remt = remote.Remote(host='somehost',
                                 user='',
                                 port=22,
                                 password='',
                                 key_file='')

        #: Remote machine class.
        self.remote = remt

        super(Remote, self).__init__(parent)
        self.init_ui()

    def init_ui(self):
        """
        Prepares the widgets tor the dialog.
        """

        self.grid = QtGui.QFormLayout(self)

        info = "The address of the remote machine or the IP"
        self.lbl_host = QtGui.QLabel('Host')
        self.lbl_host.setWhatsThis(info)
        self.lbl_host.setToolTip(info)
        self.le_host = QtGui.QLineEdit()
        self.le_host.setWhatsThis(info)
        self.le_host.setToolTip(info)
        self.le_host.setText(self.remote.host)
        self.grid.addRow(self.lbl_host, self.le_host)

        info = "The port to connect to on remote machine."
        self.lbl_port = QtGui.QLabel('Port')
        self.lbl_port.setWhatsThis(info)
        self.lbl_port.setToolTip(info)
        self.le_port = QtGui.QSpinBox()
        self.le_port.setWhatsThis(info)
        self.le_port.setToolTip(info)
        self.le_port.setValue(self.remote.port)
        self.le_port.setMaximum(65565)
        self.le_port.setMinimum(1)
        self.grid.addRow(self.lbl_port, self.le_port)

        info = "User name to use when logging in remote machine"
        self.lbl_uname = QtGui.QLabel('User name')
        self.lbl_uname.setWhatsThis(info)
        self.lbl_uname.setToolTip(info)
        self.le_uname = QtGui.QLineEdit()
        self.le_uname.setWhatsThis(info)
        self.le_uname.setToolTip(info)
        self.le_uname.setText(self.remote.user)
        self.grid.addRow(self.lbl_uname, self.le_uname)

        info = "The password associated with that user anme"
        self.lbl_pass = QtGui.QLabel('Password')
        self.lbl_pass.setWhatsThis(info)
        self.lbl_pass.setToolTip(info)
        self.le_pass = QtGui.QLineEdit()
        self.le_pass.setWhatsThis(info)
        self.le_pass.setToolTip(info)
        self.le_pass.setText(self.remote.password)
        self.le_pass.setEchoMode(QtGui.QLineEdit.Password)
        self.grid.addRow(self.lbl_pass, self.le_pass)

        info = "Path to identity file (optional)"
        self.lbl_keyf = QtGui.QLabel('Key file')
        self.lbl_keyf.setWhatsThis(info)
        self.lbl_keyf.setToolTip(info)
        self.le_keyf = QtGui.QLineEdit()
        self.le_keyf.setWhatsThis(info)
        self.le_keyf.setToolTip(info)
        self.le_keyf.setText(self.remote.key_file)
        self.grid.addRow(self.lbl_keyf, self.le_keyf)

        spat = QtGui.QSpacerItem(10, 10,
                                 QtGui.QSizePolicy.Minimum,
                                 QtGui.QSizePolicy.Expanding)
        self.grid.addItem(spat)

        # OK and Cancel buttons
        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.grid.addRow(buttons)

    def get(self):
        """
        Retreive the options user seleted.
        """
        self.remote.host = self.le_host.text()
        self.remote.user = self.le_uname.text()
        self.remote.port = self.le_port.value()
        self.remote.password = self.le_pass.text()
        self.remote.key_file = self.le_pass.text()
        return self.remote


def main():
    """
    Test app to run from command line.
    """

    #COMPANY = "pyl2extra"
    #DOMAIN = "pyl2extra.org"
    app_name = "Remote connection"

    app = QtGui.QApplication(sys.argv)
    #app.setWindowIcon(QtGui.QIcon(":/icon.png"))

    main_win = Remote()
    main_win.setWindowTitle(app_name)
    if main_win.exec_():
        print main_win.get()

if __name__ == '__main__':
    main()
