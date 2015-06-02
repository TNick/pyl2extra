#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The application allows you to show the weights of a class
that inherits from pylearn2's Model.

To start, goto File > Open pickled model... and select the
file that has your saved mode. Loading might take a while
and the window will look frozen while loading.

Once the model was loaded the parameters are presented in the
list on the left side. Select an entry to view the content for
that parameter.

For uni-dimensional values the program shows a simple,
histogram-like graph. For values with two dimensions and
more an image is presented.

If the value has more than two dimensions you can choose which
dimensions are shown using the combo-boxed that are shown in the
second vertical area next to list ov parameters.

The values shown in the image can be customized by adjusting
the treshold and chenging the way values are mapped to colors.

Examples:

    # simply run the program
    python show_weights.py

    # run the app in debug mode
    python show_weights.py --debug

    # Load a file at start-up
    python show_weights.py file.pkl

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
from pyl2extra.gui.image_viewer import SettingsMixin

COMPANY = "pyl2extra"
DOMAIN = "pyl2extra.org"
APPNAME = "PYL2 Model Browser"

def cmd_gui(args):
    """
    Shows the main GUI interface
    """
    from pyl2extra.gui.show_weights.main_window import MainWindow

    from PyQt4 import QtGui
    app = QtGui.QApplication(sys.argv)
    SettingsMixin.appSettings(DOMAIN, COMPANY, APPNAME)

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
    parser.add_argument('--model', default='')
    args = parser.parse_args()

    # prepare logging
    setup_logging(args)
    logging.debug("Application starting...")

    # run based on request
    cmd_gui(args)
    logging.debug("Application ended")


if __name__ == '__main__':
    main()
