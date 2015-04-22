#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Examples:

    # run the debugger in gui mode
    python debugger.py --debug gui
    
    # run the debugger in remote mode
    python debugger.py --debug remote --rport 5955 --pport 5956 --address "*"
    
@author: Nicu Tofan <nicu.tofan@gmail.com>
"""
import sys
import json
import os
import tempfile
import argparse
import logging


logger = None

# ----------------------------------------------------------------------------

class CustomConsoleHandler(logging.StreamHandler):
    """
    Catches the logging output and redirects it to a text box.
    """
    def __init__(self, textctrl):
        """
        Constructor.
        """
        logging.StreamHandler.__init__(self)
        self.textctrl = textctrl

    def emit(self, record):
        """
        Notified about messages.
        """
        msg = self.format(record)
        self.textctrl.append(msg)
        self.flush()

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
        env_key='PYL2E_DEBUGGER_LOG_CFG'
        log_cfg_file = os.getenv(env_key, None)
        if log_cfg_file is None:
            log_cfg_file = os.path.join(os.path.split(__file__)[0], 
                                        'logging.json')
    
    # read logging options
    custom_settings = os.path.exists(log_cfg_file)
    if custom_settings:
        with open(log_cfg_file, 'rt') as f:
            config = json.load(f)
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
    from pyl2extra.gui.debugger.main_window import MainWindow
    
    from PyQt4 import QtGui
    app = QtGui.QApplication(sys.argv)
    
    ex = MainWindow()
    ex.show()

    # The handler that prints to console
    # TODO: move in main window
    txt_handler = CustomConsoleHandler(ex.console)
    txt_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(txt_handler)

    if args.yaml:
        ex.debugger.load_file(args.yaml)
    
    logger.debug("Application started")
    sys.exit(app.exec_())
    
# ----------------------------------------------------------------------------

def cmd_remote(args):
    """
    Runs the script on a remote machne and publishes the results.
    """
    global logger
    from pyl2extra.gui.debugger.debugger_proxy import DebuggerPublisher
        
    from PyQt4 import QtCore
    app = QtCore.QCoreApplication(sys.argv)
    
    pub = DebuggerPublisher(address=args.address, 
                            req_port=args.rport, 
                            pub_port=args.pport)
    if not args.yaml:
        raise ValueError('For remote mode the yaml argument is mandatory')

    # load the file then wait for further instructions
    pub.debugger.load_file(args.yaml)
    pub.run()
    app.exit()

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
    parser.add_argument('--yaml',
                        type=str,
                        help='The YAML file to load in debugger', 
                        default=None)
                        
    subparsers = parser.add_subparsers(help='Available sub-commands')

    parser_a = subparsers.add_parser('gui', help='run in GUI mode')
    parser_a.set_defaults(func=cmd_gui)

    parser_a = subparsers.add_parser('remote', help='Run in remote mode')
    parser_a.add_argument('--rport', type=int, 
                          help='port used for requests', 
                          default='5955')
    parser_a.add_argument('--pport', type=int, 
                          help='port used for publishing', 
                          default='5956')
    parser_a.add_argument('--address', type=str, 
                          help='the address where we publish', 
                          default='*')
    parser_a.set_defaults(func=cmd_remote)

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
    args.func(args)
    logger.debug("Application ended")
    

if __name__ == '__main__':
    main()
