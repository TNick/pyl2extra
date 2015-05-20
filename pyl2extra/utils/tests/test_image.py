"""
Tests for image module in pyl2extra.utils.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import unittest
import Image, ImageDraw

from pyl2extra.utils.image import *

def create_rgb_image(width=256, height=256):
    """
    Create a simple image.

    Bottom right corner has a triangle.
    """
    image = Image.new('RGB', (width, height), (128, 0, 64))
    draw = ImageDraw.Draw(image)
    draw.polygon([(width/2, height-1), (width-1, height-1),
                  (width-1, height/2), (width/2, height-1)],
                 fill=(64, 0, 128))
    return image
    
class TestGeneric(unittest.TestCase):
    """
    Tests for slice_count().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        pass

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_seq_all(self):
        """
        Check next_seq_all()
        """
        pass
        

if __name__ == '__main__':
    unittest.main()
