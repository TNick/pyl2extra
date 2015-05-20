#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main window for the debugger.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import logging
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from PyQt4 import QtGui, QtCore

from pyl2extra.gui.guihelpers import center
from pyl2extra.gui.guihelpers import make_act

from pyl2extra.gui.dataset_window import DatasetWindow
from pyl2extra.gui.variable_window import VariableWindow
from pyl2extra.gui.object_tree_window import ObjectTreeWindow
from pyl2extra.gui.remote_window import RemoteDialog
from pyl2extra.gui.debugger_proxy import DebuggerProxy

from pyl2extra.gui import debugger
from pyl2extra.gui.signal_handler import OnlineDebugger

logger = logging.getLogger(__name__)

MSG_NO_FILE = 'No file loaded'
MSG_STS_GROUND = 'Waiting for a file...'
MSG_STS_RUNNING = 'The debugger is running. You can schedule it to pause or to stop at the end of current epoch.'
MSG_STS_PAUSED = 'The debugger is paused.'
MSG_STS_STOPPED = 'The debugger is stopped. ' + \
                  'You can start debugging the file or ' + \
                  'you can unload the file.'
MSG_STS_WAIT_PAUSE = 'The debuger is running and it will pause at the end of current epoch...'
MSG_STS_WAIT_STOP = 'The debuger is running and it will stop at the end of current epoch...'
MSG_STS_WAIT_LOAD = 'The debuger is loading the file...'

class MainWindow(QtGui.QMainWindow):
    """
    Main application window.
    """
    def __init__(self):
        """
        Constructor.
        """
        super(MainWindow, self).__init__()

        # the YAML file
        self.last_browsed_dir = None

        self.object_tree_dialogs = []
        self.dataset_windows = []
        self.var_windows = []

        self.debugger = None
        self.remote_debugger = None
        self.debugger_tree_widget = None
        self.remote_debugger_is_alive = False

        self.init_actions()
        self.init_ui()

        self.connect_debugger(OnlineDebugger())

    def init_ui(self):
        """
        Prepare GUI for main window.
        """
        self.resize(800, 600)
        center(self)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle('PyLearn2 debugger')

        lbl_debugger_state = QtGui.QLabel('State')
        lbl_statistics = QtGui.QLabel('Summary')
        lbl_file = QtGui.QLabel('File')
        lbl_commands = QtGui.QLabel('Commands')
        lbl_log = QtGui.QLabel('Log')

        self.ed_debugger_state = QtGui.QLabel(MSG_STS_GROUND)
        self.ed_statistics = QtGui.QLabel('')
        self.ed_file = QtGui.QLabel(MSG_NO_FILE)
        self.console = QtGui.QTextEdit()
        self.console.setReadOnly(True)
        
        btn_view_tree = QtGui.QPushButton('Raw')
        self.connect(btn_view_tree, QtCore.SIGNAL('clicked()'),
                     self.show_debug_train_raw)
        
        btn_view_3D = QtGui.QPushButton('Network')
        self.connect(btn_view_3D, QtCore.SIGNAL('clicked()'),
                     self.show_network_3d)        
        
        btn_view_channels = QtGui.QPushButton('Channels')
        self.connect(btn_view_channels, QtCore.SIGNAL('clicked()'),
                     self.show_debug_channels)     
        
        all_left_rows = 3

        gridw = QtGui.QWidget()
        grid = QtGui.QGridLayout()
        grid.setSpacing(5)

        grid.addWidget(lbl_debugger_state, 1, 0)
        grid.addWidget(self.ed_debugger_state, 1, 1, 1, all_left_rows)

        grid.addWidget(lbl_statistics, 2, 0)
        grid.addWidget(self.ed_statistics, 2, 1, 2, all_left_rows)

        grid.addWidget(lbl_file, 3, 0)
        grid.addWidget(self.ed_file, 3, 1, 3, all_left_rows)
        
        grid.addWidget(lbl_commands, 4, 0)
        grid.addWidget(btn_view_tree, 4, 1)
        grid.addWidget(btn_view_3D, 4, 2)
        grid.addWidget(btn_view_channels, 4, 3)
        
        grid.addWidget(lbl_log, 5, 0)
        grid.addWidget(self.console, 5, 1, 16, all_left_rows)

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
        self.act_load_yaml = make_act('Open &YAML ...', self,
                                      'folders_explorer.png',
                                      'Ctrl+O',
                                      'Load YAML File',
                                      self.browse_yaml)
        self.act_load_pkl = make_act('Open P&KL ...', self,
                                     'folder_table.png',
                                     'Ctrl+K',
                                     'Load PKL File',
                                     self.browse_pkl)
        self.act_load_dset = make_act('Open &Dataset ...', self,
                                      'folder_database.png',
                                      'Ctrl+D',
                                      'Load Dataset File',
                                      self.browse_dataset)

        self.act_run_end = make_act('Continue', self,
                                    'control_play_blue.png',
                                    'F5',
                                    'Run until the train ends')
        self.act_run_end.setEnabled(False)
        self.act_run_epoch = make_act('Run an epoch', self,
                                      'control_cursor_blue.png',
                                      'F10',
                                      'Runs an epoch then stop again')
        self.act_run_epoch.setEnabled(False)
        self.act_run_stop = make_act('Stop', self,
                                     'control_stop_blue.png',
                                     'F5',
                                     'Terminate current training',
                                     self.stop_or_pause)
        self.act_run_stop.setEnabled(False)
        self.act_run_pause = make_act('Pause', self,
                                      'control_pause_blue.png',
                                      'F5',
                                      'Pause a running job at next epoch',
                                     self.stop_or_pause)
        self.act_run_pause.setEnabled(False)
        self.act_run_load = make_act('Debug a YAML file...', self,
                                     'folder_go.png',
                                     'Ctrl+W',
                                     'Load a file for debugging',
                                     self.browse_for_debug)
        self.act_run_load.setEnabled(True)
        self.act_run_unload = make_act('Unload debugged file', self,
                                       'control_eject_blue.png',
                                       'Ctrl+Shift+W',
                                       'Unload current debugged file')
        self.act_run_unload.setEnabled(False)
        self.act_run_remote = make_act('Debug remote...', self,
                                       'ftp.png',
                                       'Ctrl+Shift+R',
                                       'Allows debugging a process '
                                       'running on a remote machine',
                                       self.show_remote_debug)
        self.act_run_remote.setEnabled(True)


        menubar = self.menuBar()

        menu_file = menubar.addMenu('&File')
        menu_file.addAction(self.act_load_yaml)
        menu_file.addAction(self.act_load_pkl)
        menu_file.addAction(self.act_load_dset)
        menu_file.addAction(self.act_exit)

        menu_debug = menubar.addMenu('&Debug')
        menu_debug.addAction(self.act_run_load)
        menu_debug.addAction(self.act_run_remote)
        menu_debug.addAction(self.act_run_unload)
        menu_debug.addSeparator()
        menu_debug.addAction(self.act_run_end)
        menu_debug.addAction(self.act_run_epoch)
        menu_debug.addAction(self.act_run_pause)
        menu_debug.addAction(self.act_run_stop)

        self.toolbar = self.addToolBar('General')
        self.toolbar.addAction(self.act_load_yaml)
        self.toolbar.addAction(self.act_load_pkl)
        self.toolbar.addAction(self.act_load_dset)
        self.toolbar.addAction(self.act_exit)

        self.toolbar = self.addToolBar('Debug')
        self.toolbar.addAction(self.act_run_load)
        self.toolbar.addAction(self.act_run_remote)
        self.toolbar.addAction(self.act_run_unload)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_run_end)
        self.toolbar.addAction(self.act_run_epoch)
        self.toolbar.addAction(self.act_run_pause)
        self.toolbar.addAction(self.act_run_stop)


    def closeEvent(self, event):
        """
        Build-in close event.
        """
        reply = QtGui.QMessageBox.question(self, 'Message',
                                           "Are you sure you want to quit?",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
                                           QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            event.accept()
            QtCore.QCoreApplication.exit(0)
        else:
            event.ignore()

    def collect_show_statistics(self):
        """
        Collects the statistics from the debugger and presens them in GUI.
        
        TODO: implement
        """
        epocs = 0
        examples = 0
        channels = 0
        learning_rate = 0.0
        self.show_statistics(epocs=epocs,
                             examples=examples, 
                             channels=channels,
                             learning_rate=learning_rate)

    def show_statistics(self, epocs=0, examples=0, 
                        channels=0, learning_rate=0.0):
        """
        Present the statistics.
        """
        data = (epocs, examples, channels, learning_rate)
        self.ed_statistics.setText('%d epochs, %d examples, '
                                   '%d channels, learning rate: %f ' % data)


    def show_debug_train_raw(self):
        """
        Shows the raw content of the train object.
        
        TODO: implement
        """
        pass
    
    def show_network_3d(self):
        """
        Shows the raw content of the train object.
        
        TODO: implement
        """
        pass

    def show_debug_channels(self):
        """
        Shows the plot of the channels
        
        TODO: implement
        """
        pass

    def set_state_text(self, state_text):
        """
        Changes the state text.
        """
        self.ed_debugger_state.setText(state_text)
        self.ed_debugger_state.setToolTip(state_text)

    def load_pkl(self, fname):
        """
        Slot that loads a PKL file.
        """
        if not fname:
            return
        try:
            pkl_tree = serial.load(fname, retry=False)
            self.show_object_tree(pkl_tree)
        except Exception, exc:
            logger.error('Loading pkl file failed', exc_info=True)
            QtGui.QMessageBox.warning(self, 'Exception', str(exc))

    def load_dataset(self, fname):
        """
        Slot that loads a Dataset file.
        """
        if not fname:
            return
        try:
            dataset = serial.load(fname, retry=False)
            self.show_dataset(dataset)
        except Exception, exc:
            logger.error('Loading dataset file failed', exc_info=True)
            QtGui.QMessageBox.warning(self, 'Exception', str(exc))

    def load_yaml(self, fname):
        """
        Slot that loads a YAML file.
        """
        if not fname:
            return
        try:
            # publish environment variables relevant to this file
            serial.prepare_train_file(fname)

            # load the tree of Proxy objects
            environ = {}
            yaml_tree = yaml_parse.load_path(fname,
                                             instantiate=False,
                                             environ=environ)
            yaml_tree = yaml_parse._instantiate(yaml_tree)
            self.show_object_tree(yaml_tree)
        except Exception, exc:
            logger.error('Loading aml file failed', exc_info=True)
            QtGui.QMessageBox.warning(self, 'Exception', str(exc))

    def browse_yaml(self):
        """
        Slot that browse for and loads a YAML file.
        """
        if self.last_browsed_dir is None:
            self.last_browsed_dir = QtCore.QDir.homePath()
        fname = QtGui.QFileDialog.getOpenFileName(self,
                                                  'Open YAML file',
                                                  self.last_browsed_dir)
        if not fname:
            return
        self.load_yaml(fname)

    def browse_for_debug(self):
        """
        Slot that browse for and loads a YAML file.
        """
        if self.last_browsed_dir is None:
            self.last_browsed_dir = QtCore.QDir.homePath()
        fname = QtGui.QFileDialog.getOpenFileName(self,
                                                  'Open YAML file',
                                                  self.last_browsed_dir)
        if not fname:
            return
        self.act_run_load.setEnabled(False)
        self.act_run_remote.setEnabled(False)

        if self.remote_debugger:
            self.connect_debugger(OnlineDebugger())
        self.set_state_text(MSG_STS_WAIT_LOAD)
        self.debugger.load_file(fname)

    def browse_dataset(self):
        """
        Slot that browse for and loads a dataset file.
        """
        if self.last_browsed_dir is None:
            self.last_browsed_dir = QtCore.QDir.homePath()
        fname = QtGui.QFileDialog.getOpenFileName(self,
                                                  'Open a pickled dataset file',
                                                  self.last_browsed_dir)
        if not fname:
            return
        self.load_dataset(fname)

    def browse_pkl(self):
        """
        Slot that browse for and loads a PKL file.
        """
        if self.last_browsed_dir is None:
            self.last_browsed_dir = QtCore.QDir.homePath()
        fname = QtGui.QFileDialog.getOpenFileName(self,
                                                  'Open PKL file',
                                                  self.last_browsed_dir)
        if not fname:
            return
        self.load_pkl(fname)

    def show_dataset(self, dataset):
        """
        Open a new window and show the dataset.
        """
        if dataset is None:
            return
        dsw = DatasetWindow(self, dataset)
        dsw.show()
        self.dataset_windows.append(dsw)
        return dsw

    def show_remote_debug(self):
        """
        Parameters for remote debugging.
        """
        dlg = RemoteDialog(self)
        keep_running = True
        while keep_running:
            keep_running = dlg.exec_()
            if keep_running:
                values = dlg.get_values()
                if len(values['address']) == 0:
                    QtGui.QMessageBox.warning(self,
                                              'Error',
                                              'An address must be provided')
                else:
                    keep_running = False
                    self.debug_remote_start(values)

    def show_variable(self, var):
        """
        Open a new window and show the dataset.
        """
        if var is None:
            return
        dsw = VariableWindow(self, var)
        dsw.show()
        self.var_windows.append(dsw)
        return dsw

    def show_object_tree(self, obj_tree):
        """
        High level method to present current YAML object.
        """

        if obj_tree is None:
            return
        tree_widget = ObjectTreeWindow(self, obj_tree)
        tree_widget.show()
        self.object_tree_dialogs.append(tree_widget)
        return tree_widget

    def debug_remote_start(self, args):
        """
        Starts a debug session with the remote host.
        """
        assert args.has_key('address')
        assert args.has_key('rport')
        assert args.has_key('pport')

        self.act_run_load.setEnabled(False)
        self.act_run_remote.setEnabled(False)

        dbg_inst = DebuggerProxy(address=args['address'],
                                 req_port=args['rport'],
                                 pub_port=args['pport'])
        self.connect_debugger(dbg_inst)
        
        if dbg_inst.is_running():
            logger.info('The debugger is running')
        else:
            sts_str = debugger.state_name(dbg_inst.last_known_status)
            logger.info('Debugger status: %s' % sts_str)

    def debug_start(self, yaml_file):
        """
        Slot that connects to debugger; raised when the debugger
        loaded a file.
        """
        self.ed_file.setText(yaml_file)
        self.set_state_text(MSG_STS_STOPPED)
        self.show_statistics()
        
        self.act_run_end.setEnabled(True)
        self.act_run_epoch.setEnabled(True)
        self.act_run_stop.setEnabled(True)
        self.act_run_pause.setEnabled(False)
        self.act_run_load.setEnabled(False)
        self.act_run_remote.setEnabled(False)
        self.act_run_unload.setEnabled(True)
        
        self.statusBar().showMessage('Debugger loaded ' + yaml_file)
        self.debugger_tree_widget = self.show_object_tree(self.debugger.train_obj)
        self.debugger_tree_widget.setWindowTitle('DEBUG: ' + yaml_file)


    def debug_end(self, yaml_file):
        """
        Slot that connects to debugger; raised when the debugger
        unloaded a file.
        """
        self.ed_file.setText(MSG_NO_FILE)
        self.set_state_text(MSG_STS_GROUND)
        self.act_run_end.setEnabled(False)
        self.act_run_epoch.setEnabled(False)
        self.act_run_stop.setEnabled(False)
        self.act_run_pause.setEnabled(False)
        self.act_run_load.setEnabled(True)
        self.act_run_remote.setEnabled(True)
        self.act_run_unload.setEnabled(True)
        self.statusBar().showMessage('Debugger unloaded ' + yaml_file)
        if not self.debugger_tree_widget is None:
            self.object_tree_dialogs.remove(self.debugger_tree_widget)
            self.debugger_tree_widget.close()
            self.debugger_tree_widget = None

    def debug_run(self):
        """
        Slot that connects to debugger; raised when the debugger
        starts a run.
        """
        self.set_state_text(MSG_STS_RUNNING)
        self.act_run_end.setEnabled(False)
        self.act_run_epoch.setEnabled(False)
        self.act_run_stop.setEnabled(True)
        self.act_run_pause.setEnabled(True)
        self.act_run_unload.setEnabled(False)
        self.statusBar().showMessage('Debugger running')

    def debug_paused(self):
        """
        Slot that connects to debugger; raised when the debugger
        enters pause.
        """
        self.set_state_text(MSG_STS_PAUSED)
        self.act_run_end.setEnabled(True)
        self.act_run_epoch.setEnabled(True)
        self.act_run_stop.setEnabled(True)
        self.act_run_pause.setEnabled(False)
        self.act_run_unload.setEnabled(True)
        self.statusBar().showMessage('Debugger paused')
        
        self.collect_show_statistics()

    def debug_stopped(self):
        """
        Slot that connects to debugger; raised when the debugger
        was stopped.
        """
        self.set_state_text(MSG_STS_STOPPED)
        self.act_run_end.setEnabled(True)
        self.act_run_epoch.setEnabled(True)
        self.act_run_stop.setEnabled(False)
        self.act_run_pause.setEnabled(False)
        self.act_run_unload.setEnabled(True)
        self.statusBar().showMessage('Debugger stopped')

    def stop_or_pause(self):
        """
        The debugger will not react to stop/pause right away.
        """
        self.act_run_stop.setEnabled(False)
        self.act_run_pause.setEnabled(False)
        if self.sender() is self.act_run_stop:
            self.set_state_text(MSG_STS_WAIT_STOP)
        else:
            self.set_state_text(MSG_STS_WAIT_PAUSE)

    def debug_state_change(self, old_state, new_state):
        """
        Slot that connects to debugger; raised when the debugger
        changes the state.
        """
        sts_name = debugger.state_name(new_state)
        self.statusBar().showMessage('Debugger status: ' + sts_name)

    def debug_error(self, message):
        """
        Slot that connects to debugger; raised when the debugger
        encounters an error.
        """
        QtGui.QMessageBox.warning(self, 'Debugger error', message)
        self.statusBar().showMessage('Debugger error: ' + message)

    def connect_debugger(self, debugger, is_remote=False):
        """
        Makes a debugger the current instance.
        """
        self.disconnect_debugger()

        self.debugger_tree_widget = None
        self.remote_debugger = is_remote
        self.debugger = debugger

        # info signals
        self.connect(debugger, QtCore.SIGNAL('debug_start(QString)'),
                     self.debug_start)
        self.connect(debugger, QtCore.SIGNAL('debug_end(QString)'),
                     self.debug_end)
        self.connect(debugger, QtCore.SIGNAL('debug_run()'),
                     self.debug_run)
        self.connect(debugger, QtCore.SIGNAL('debug_paused()'),
                     self.debug_paused)
        self.connect(debugger, QtCore.SIGNAL('debug_stopped()'),
                     self.debug_stopped)
        self.connect(debugger, QtCore.SIGNAL('debug_error(QString)'),
                     self.debug_error)
        self.connect(debugger, QtCore.SIGNAL('debug_state_change(int,int)'),
                     self.debug_state_change)
        if is_remote:
            self.connect(debugger, QtCore.SIGNAL('alive(bool)'),
                         self.remote_debugger_alive)

        # control slots
        QtCore.QObject.connect(self.act_run_end,
                               QtCore.SIGNAL("triggered()"),
                               debugger.dbg_continue)
        QtCore.QObject.connect(self.act_run_epoch,
                               QtCore.SIGNAL("triggered()"),
                               debugger.dbg_run_one)
        QtCore.QObject.connect(self.act_run_stop,
                               QtCore.SIGNAL("triggered()"),
                               debugger.dbg_stop)
        QtCore.QObject.connect(self.act_run_pause,
                               QtCore.SIGNAL("triggered()"),
                               debugger.dbg_pause)
        QtCore.QObject.connect(self.act_run_unload,
                               QtCore.SIGNAL("triggered()"),
                               debugger.unload_file)


    def remote_debugger_alive(self, new_state):
        """
        Get informed if the harbeat is not received.
        """
        if new_state != self.remote_debugger_is_alive:
            if new_state:
                self.statusBar().showMessage('Remote debugger online')
            else:
                self.statusBar().showMessage('Remote debugger offline')
            self.remote_debugger_is_alive = new_state

    def disconnect_debugger(self):
        """
        Disconnects current debugger.
        """
        if not self.debugger:
            return

        self.debugger_tree_widget = None
        if self.debugger.state != debugger.STS_GROUND:
            self.debugger.unload_file()

        # info signals
        self.disconnect(self.debugger, QtCore.SIGNAL('debug_start(QString)'),
                        self.debug_start)
        self.disconnect(self.debugger, QtCore.SIGNAL('debug_end(QString)'),
                        self.debug_end)
        self.disconnect(self.debugger, QtCore.SIGNAL('debug_run()'),
                        self.debug_run)
        self.disconnect(self.debugger, QtCore.SIGNAL('debug_paused()'),
                        self.debug_paused)
        self.disconnect(self.debugger, QtCore.SIGNAL('debug_stopped()'),
                        self.debug_stopped)
        self.disconnect(self.debugger, QtCore.SIGNAL('error(QString)'),
                        self.debug_error)
        if self.remote_debugger:
            self.disconnect(debugger, QtCore.SIGNAL('alive(bool)'),
                            self.remote_debugger_alive)

        # control slots
        QtCore.QObject.disconnect(self.act_run_end,
                                  QtCore.SIGNAL("triggered()"),
                                  self.debugger.dbg_continue)
        QtCore.QObject.disconnect(self.act_run_epoch,
                                  QtCore.SIGNAL("triggered()"),
                                  self.debugger.dbg_run_one)
        QtCore.QObject.disconnect(self.act_run_stop,
                                  QtCore.SIGNAL("triggered()"),
                                  self.debugger.dbg_stop)
        QtCore.QObject.disconnect(self.act_run_pause,
                                  QtCore.SIGNAL("triggered()"),
                                  self.debugger.dbg_pause)
        QtCore.QObject.disconnect(self.act_run_unload,
                                  QtCore.SIGNAL("triggered()"),
                                  self.debugger.unload_file)

        self.debugger = None
        self.remote_debugger_is_alive = False
