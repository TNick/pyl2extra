#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A dialog that allows the user to browse the files and directories on
a remote machine.

It provides a GUI to :class:`pyl2extra.utils.remote.Remote`.
"""
from __future__ import with_statement

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import os
from PyQt4 import (QtCore, QtGui)
import sys

from pyl2extra.utils import remote
from pyl2extra.gui.guihelpers import center
from pyl2extra.gui.remote import Remote


class RemoteBrowse(QtGui.QDialog):
    """
    Dialog to allow exploring a remote host.

    Parameters
    ----------
    parent : str, optional
        Parent widget.
    remt : str, optional
        The remote in question.
    """
    def __init__(self, parent=None, remt=None, local_path=None, 
                 settings=None):

        if settings is None:        
            self.settings = QtCore.QSettings('pyl2extra', 'remote_browse')
        else:
            self.settings = settings
        
        if remt is None:
            remt = remote.Remote(host='somehost',
                                 user='',
                                 port=22,
                                 password='',
                                 key_file='')
        if local_path is None:
            local_path = os.getcwd()

        #: Remote machine class.
        self.remote = None
        #: Path on local machine.
        self.local_path = local_path

        super(RemoteBrowse, self).__init__(parent)
        self.init_ui()
        
        self.change_remote(remt)
        self.change_local_path(local_path)
        self.local_tree.resizeColumnToContents(0)

    def init_ui(self):
        """
        Prepares the widgets tor the dialog.
        """
        try:
            size = self.settings.value('RemoteBrowse/size', type=QtCore.QSize)
            self.resize(size)
        except TypeError:
            self.resize(900, 800)
            
        try:
            point = self.settings.value('RemoteBrowse/pos', type=QtCore.QPoint)
            self.move(point)
        except TypeError:
            center(self)

        self.grid = QtGui.QGridLayout(self)
        
        info = "The path on local machine"
        self.lbl_local = QtGui.QLabel('Local path')
        self.lbl_local.setWhatsThis(info)
        self.lbl_local.setToolTip(info)
        self.grid.addWidget(self.lbl_local, 0, 0, 1, 1)    
        self.le_local = QtGui.QLineEdit()
        self.le_local.setWhatsThis(info)
        self.le_local.setToolTip(info)
        self.le_local.setText(self.local_path)
        self.le_local.editingFinished.connect(self._local_select)
        self.grid.addWidget(self.le_local, 0, 1, 1, 1)        
        
        info = "Files and folders on local machine"
        self.fsmodel = QtGui.QFileSystemModel()
        self.fsmodel.setRootPath(self.local_path)
        self.local_tree = QtGui.QTreeView()
        self.local_tree.setModel(self.fsmodel)
        self.local_tree.setWhatsThis(info)
        self.local_tree.setToolTip(info)
        self.local_tree.selectionModel().currentChanged.connect(
            self._local_item_changed)
        self.grid.addWidget(self.local_tree, 1, 0, 1, 2)
        
        
        self.mid_grid = QtGui.QVBoxLayout()
        
        info = "Choose the remote machine to explore"
        self.b_remote = QtGui.QToolButton()
        self.b_remote.setWhatsThis(info)
        self.b_remote.setToolTip(info)
        self.b_remote.clicked.connect(self._choose_remote)
        self.mid_grid.addWidget(self.b_remote)
        
        
        info = "Copy from local machine to remote machine"
        self.b_get = QtGui.QToolButton()
        self.b_get.setWhatsThis(info)
        self.b_get.setToolTip(info)
        self.b_remote.clicked.connect(self._get_file)
        self.mid_grid.addWidget(self.b_get)  
        
        info = "Copy from remote machine to local machine"
        self.b_put = QtGui.QToolButton()
        self.b_put.setWhatsThis(info)
        self.b_put.setToolTip(info)
        self.b_remote.clicked.connect(self._put_file)
        self.mid_grid.addWidget(self.b_put)  
        
        spat = QtGui.QSpacerItem(10, 10,
                                 QtGui.QSizePolicy.Minimum,
                                 QtGui.QSizePolicy.Expanding)
        self.mid_grid.addItem(spat)
        
        self.grid.addLayout(self.mid_grid, 0, 2, 2, 1)
        
        
        info = "The path on remote machine"
        self.lbl_remote = QtGui.QLabel('Remote path')
        self.lbl_remote.setWhatsThis(info)
        self.lbl_remote.setToolTip(info)
        self.grid.addWidget(self.lbl_remote, 0, 3, 1, 1)    
        self.le_remote = QtGui.QLineEdit()
        self.le_remote.setWhatsThis(info)
        self.le_remote.setToolTip(info)
        self.le_remote.setText('')
        self.grid.addWidget(self.le_remote, 0, 4, 1, 1)        
        
        info = "Files and folders on remote machine"
        self.remote_tree = QtGui.QTreeWidget()
        self.remote_tree.setWhatsThis(info)
        self.remote_tree.setToolTip(info)
        self.remote_tree.currentItemChanged.connect(self._remote_item_changed)
        self.remote_tree.setHeaderLabels(['Name', 'Type', 'Size', 
                                          'Date modified'])
        self.grid.addWidget(self.remote_tree, 1, 3, 1, 2)
        
    def _choose_remote(self):
        """
        Slot allowing to change the remote machine.
        """
        main_win = Remote()
        main_win.setWindowTitle('Choose the connection parameters')
        if main_win.exec_():
            self.remote = main_win.get()
            self.change_remote(main_win.get())
        
    def _get_file(self):
        """
        Slot allowing to copy a file from the remote machine.
        """
        pass

    def _put_file(self):
        """
        Slot allowing to copy a file to the remote machine.
        """
        pass

    def _local_item_changed(self, current, previous):
        """
        Slot informed that current file changed.
        """
        if current is None:
            self.le_local.setText('')
        else:
            self.le_local.setText(self.fsmodel.filePath(current))

    def _local_select(self):
        """
        Change current item in local tree.
        """
        self.change_local_path(self.le_local.text())
        
    def _remote_item_changed(self, current, previous):
        """
        Slot informed that current file changed.
        """
        #if current is None:
        #    self.le_remote.setText('')
        #else:
        #    self.le_remote.setText(self.fsmodel.filePath(current))        
        
    def closeEvent(self, event):
        """
        Build-in close event.
        """
        self.settings.setValue('RemoteBrowse/pos', self.pos())
        self.settings.setValue('RemoteBrowse/size', self.size())

        del self.settings
        event.accept()

    def change_remote(self, remote):
        """
        Change the remote that we're browsing.
        """
        if not self.remote is None:
            self.remote.deactivate()
        self.remote = remote
        if not self.remote is None:
            self.remote.activate()
            
        b_ok = self.remote.test_connection()
        self.b_get.setEnabled(b_ok)
        self.b_put.setEnabled(b_ok)
        self.lbl_remote.setEnabled(b_ok)
        self.le_remote.setEnabled(b_ok)
        self.remote_tree.setEnabled(b_ok)
        
        if b_ok:
            self.change_remote_path(self.remote.cwd)
        else:
            self.le_remote.setText('')    
            
    def change_remote_path(self, rpath):
        """
        Change the remote path that we're seeing.
        """
        self.remote.cwd = rpath
        self.le_remote.setText(rpath)
        
            
    def change_local_path(self, lpath):
        """
        Change the remote path that we're seeing.
        """
        midx = self.fsmodel.index(lpath)
        if midx.isValid():
            self.local_tree.setCurrentIndex(midx)     
            self.local_tree.scrollTo(midx)


def main():
    """
    Test app to run from command line.
    """

    #COMPANY = "pyl2extra"
    #DOMAIN = "pyl2extra.org"
    app_name = "Remote Explorer"

    app = QtGui.QApplication(sys.argv)
    #app.setWindowIcon(QtGui.QIcon(":/icon.png"))

    main_win = RemoteBrowse()
    main_win.setWindowTitle(app_name)
    main_win.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
