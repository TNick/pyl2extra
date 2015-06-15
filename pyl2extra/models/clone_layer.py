# -*- coding: utf-8 -*-
"""
Clone la layer's characteristics as weights and biases.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"


class Stat(object):
    """
    Information about a file.
    """
    def __init__(self, model):
        self.model = model
    
        
