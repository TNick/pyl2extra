#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interface for signal handler used by the debugger and implementations.

To avoid adding a dependency to Qt on the debugger class, the signals are
handled by a class implementing the SignalHandler interface.
The debugger may be used in headless deployments without PyQt installed
and it may relay the information to a machine running the GUI part.

If PyQt is present the module also provides a QObject implementation.

@author: Nicu Tofan <nicu.tofan@gmail.com>
"""
import logging
from learn_spot.gui.debugger import Debugger

logger = logging.getLogger(__name__)

try:
    from PyQt4 import QtCore
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


class SignalHandler(object):
    """
    Interface for the class used by the Debugger to communicate.
    """
    def __init__(self):
        """
        Constructor.
        """
        super(SignalHandler, self).__init__()

    def debug_start(self, yaml_file):
        """
        Slot that connects to debugger; raised when the debugger
        loaded a file.
        """
        raise NotImplementedError

    def debug_paused(self):
        """
        Slot that connects to debugger; raised when the debugger
        enters pause.
        """
        raise NotImplementedError

    def debug_stopped(self):
        """
        Slot that connects to debugger; raised when the debugger
        was stopped.
        """
        raise NotImplementedError

    def debug_run(self):
        """
        Slot that connects to debugger; raised when the debugger
        starts a run.
        """
        raise NotImplementedError

    def debug_end(self, yaml_file):
        """
        Slot that connects to debugger; raised when the debugger
        unloaded a file.
        """
        raise NotImplementedError

    def debug_error(self, message):
        """
        Slot that connects to debugger; raised when the debugger
        has an error to report.
        """
        raise NotImplementedError

    def debug_state_change(self, old_state, new_state):
        """
        Slot that connects to debugger; raised when the debugger
        changes the state.
        """
        raise NotImplementedError



if QT_AVAILABLE:

    ERR_SIGNAL = QtCore.SIGNAL("error(QString)")

    class OnlineDebugger(Debugger, SignalHandler, QtCore.QObject):
        """
        The debugger used on a machine that has both the test
        environment and the graphical interface.
        """
        def __init__(self):
            """
            Constructor.
            """
            super(OnlineDebugger, self).__init__()
            self.signal_handler = self

        def debug_start(self, yaml_file):
            """
            Slot that connects to debugger; raised when the debugger
            loaded a file.
            """
            self.emit(QtCore.SIGNAL("debug_start(QString)"), yaml_file)

        def debug_paused(self):
            """
            Slot that connects to debugger; raised when the debugger
            enters pause.
            """
            self.emit(QtCore.SIGNAL("debug_paused"))

        def debug_stopped(self):
            """
            Slot that connects to debugger; raised when the debugger
            was stopped.
            """
            self.emit(QtCore.SIGNAL("debug_stopped"))

        def debug_run(self):
            """
            Slot that connects to debugger; raised when the debugger
            starts a run.
            """
            self.emit(QtCore.SIGNAL("debug_run"))

        def debug_end(self, yaml_file):
            """
            Slot that connects to debugger; raised when the debugger
            unloaded a file.
            """
            self.emit(QtCore.SIGNAL("debug_end(QString)"), yaml_file)

        def debug_error(self, message):
            """
            Slot that connects to debugger; raised when the debugger
            has an error to report.
            """
            self.emit(QtCore.SIGNAL("debug_error(QString)"), message)

        def debug_state_change(self, old_state, new_state):
            """
            Slot that connects to debugger; raised when the debugger
            changes the state.
            """
            self.emit(QtCore.SIGNAL("debug_state_change(int,int)"), old_state, new_state)
