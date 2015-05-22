"""
Tests for helper functions related to numpy.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import numpy
from pyl2extra.utils.npy import slice_1d, slice_2d, slice_3d
import unittest

class TestSlice(unittest.TestCase):
    """
    Tests for ImgDataset
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.ary = numpy.array(range(2*3*4*5*6)).reshape(2, 3, 4, 5, 6)
        self.cnt = 0

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_1d(self):
        """
        Slice an array in 1D pices
        """
        prev = -6
        for (address, pice) in slice_1d(self.ary):
            self.cnt = self.cnt + 1
            self.assertEqual(len(pice), 6)
            self.assertEqual(len(address), 4)
            self.assertEqual(pice[0], prev+6)
            prev = pice[0]
        self.assertEqual(self.cnt, 2*3*4*5)

    def test_2d(self):
        """
        Slice an array in 2D pices
        """
        prev = -(6*5)
        for (address, pice) in slice_2d(self.ary):
            self.cnt = self.cnt + 1
            self.assertEqual(len(pice.shape), 2)
            self.assertEqual(pice.shape[0], 5)
            self.assertEqual(pice.shape[1], 6)
            self.assertEqual(len(address), 3)
            self.assertEqual(pice[0, 0], prev+(6*5))
            prev = pice[0, 0]
        self.assertEqual(self.cnt, 2*3*4)

    def test_3d(self):
        """
        Slice an array in 3D pices
        """
        prev = -(6*5*4)
        for (address, pice) in slice_3d(self.ary):
            self.cnt = self.cnt + 1
            self.assertEqual(len(pice.shape), 3)
            self.assertEqual(pice.shape[0], 4)
            self.assertEqual(pice.shape[1], 5)
            self.assertEqual(pice.shape[2], 6)
            self.assertEqual(len(address), 2)
            self.assertEqual(pice[0, 0, 0], prev+(6*5*4))
            prev = pice[0, 0, 0]
        self.assertEqual(self.cnt, 2*3)

if __name__ == '__main__':
    unittest.main(argv=['--verbose'])
