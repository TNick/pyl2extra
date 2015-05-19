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

logger = None

# ----------------------------------------------------------------------------

def setup_logging(args,
                  default_level=logging.DEBUG):
    """
    Setup logging configuration
    """
    global logger

    # get a path for logging config
    if not args.log_cfg is None:
        log_cfg_file = args.log_cfg
    else:
        env_key = 'PYL2E_DEBUGGER_LOG_CFG'
        log_cfg_file = os.getenv(env_key, None)
        if log_cfg_file is None:
            log_cfg_file = os.path.join(os.path.split(__file__)[0],
                                        'logging.json')

    # read logging options
    custom_settings = os.path.exists(log_cfg_file)
    if custom_settings:
        with open(log_cfg_file, 'rt') as fhand:
            config = json.load(fhand)
        logging.config.dictConfig(config,
                                  disable_existing_loggers=False)
    else:
        logging.basicConfig(level=default_level)

    logger = logging.getLogger()
    if args.verbose_logging:
        raise NotImplementedError('verbose_logging not implemented, yet')
    if args.timestamp:
        raise NotImplementedError('verbose_logging not implemented, yet')

    # Set the root logger level.
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # if the user requested a log file give her one
    if not custom_settings or args.log:
        # The handler that logs to a file
        if args.log:
            log_path = args.log
        else:
            log_path = tempfile.mktemp(prefix='pylearn2-debugger-',
                                       suffix='.json')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Debugger for pylearn2 models.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--level-name', '-L',
                        action='store_true',
                        help='Display the log level (e.g. DEBUG, INFO) '
                             'for each logged message')
    parser.add_argument('--timestamp', '-T',
                        action='store_true',
                        help='Display human-readable timestamps for '
                             'each logged message')
    parser.add_argument('--verbose-logging', '-V',
                        action='store_true',
                        help='Display timestamp, log level and source '
                             'logger for every logged message '
                             '(implies -T).')
    parser.add_argument('--debug', '-D',
                        action='store_true',
                        help='Display any DEBUG-level log messages, '
                             'suppressed by default.')
    parser.add_argument('--log',
                        type=str,
                        help='The log file.',
                        default=None)
    parser.add_argument('--log-cfg',
                        type=str,
                        help='The log config file in json format.',
                        default=None)

    return parser

# ----------------------------------------------------------------------------

def main():
    """
    Module entry point.
    """
    global logger

    # look at the arguments
    parser = make_argument_parser()
    args = parser.parse_args()

    # prepare logging
    setup_logging(args)
    logger.debug("Application starting...")

    # run based on request
    cmd_gui(args)
    logger.debug("Application ended")


if __name__ == '__main__':
    main()
