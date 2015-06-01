#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Examples:

    # simply run the program
    python show_weights.py

    # run the app in debug mode
    python show_weights.py --debug

"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import sys
import logging

from pyl2extra.utils.script import setup_logging, make_argument_parser

def cmd_gui(args):
    """
    Shows the main GUI interface
    """
    from pyl2extra.gui.show_weights.main_window import MainWindow

    from PyQt4 import QtGui
    app = QtGui.QApplication(sys.argv)

    ex = MainWindow()
    ex.show()

    if len(args.model) > 0:
        ex.load_model_file(args.model)

    logging.debug("Application started")
    sys.exit(app.exec_())

def main():
    """
    Module entry point.
    """

    # look at the arguments
    parser = make_argument_parser("Debugger for pylearn2 models.")
    parser.add_argument('model', default='')
    args = parser.parse_args()

    # prepare logging
    setup_logging(args)
    logging.debug("Application starting...")

    # run based on request
    cmd_gui(args)
    logging.debug("Application ended")


if __name__ == '__main__':
    main()
