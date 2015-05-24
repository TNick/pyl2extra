#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Examples:

    # simply run the program
    python viewprep.py

    # run run the app in debug mode
    python viewprep.py --debug

"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import sys
import json
import os
import tempfile
import argparse
import logging

from pyl2extra.utils.script import setup_logging, make_argument_parser

def cmd_gui(args):
    """
    Shows the main GUI interface
    """
    global logger
    from pyl2extra.gui.viewprep.main_window import MainWindow

    from PyQt4 import QtGui
    app = QtGui.QApplication(sys.argv)

    ex = MainWindow()
    ex.show()

    logger.debug("Application started")
    sys.exit(app.exec_())

def main():
    """
    Module entry point.
    """
    global logger

    # look at the arguments
    parser = make_argument_parser("Debugger for pylearn2 models.")
    args = parser.parse_args()

    # prepare logging
    setup_logging(args)
    logger.debug("Application starting...")

    # run based on request
    cmd_gui(args)
    logger.debug("Application ended")


if __name__ == '__main__':
    main()
