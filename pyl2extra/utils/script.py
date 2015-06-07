"""
Helper functions for scripts.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import argparse
import json
import logging
import os
import tempfile


def setup_logging(args, logger=None, default_level=logging.DEBUG):
    """
    Setup logging configuration
    """
    if logger is None:
        logger = logging.getLogger()

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

    if args.verbose_logging:
        raise NotImplementedError('verbose_logging not implemented, yet')
    if args.timestamp:
        raise NotImplementedError('timestamp not implemented, yet')

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
        if args.debug:
            file_handler.setLevel(logging.DEBUG)
        else:
            file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - '
                                      '%(name)s - '
                                      '%(levelname)s - '
                                      '%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def make_argument_parser(description):
    """
    Creates an ArgumentParser to read the options for script from sys.argv.
    """
    parser = argparse.ArgumentParser(
        description=description,
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

