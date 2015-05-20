"""
Custom logger handle that adds the output to a text window..
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"


import logging

class LoggerToWidget(logging.StreamHandler):
    """
    Catches the logging output and redirects it to a text box.

    Parameters
    ----------
    textctrl : widget
        An instance that implements an ``append()`` method with a single
        argument (the text to show).
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
