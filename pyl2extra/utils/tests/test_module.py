"""
Tests for the functions in the utils module.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import unittest

from pyl2extra.utils import slice_count


class TestSliceCount(unittest.TestCase):
    """
    Tests for slice_count().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        pass

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_all(self):
        """
        Check slice_count()
        """
        slc = slice(0, 0)
        self.assertEqual(slice_count(slc), 0)

        slc = slice(0, 1)
        self.assertEqual(slice_count(slc), 1)

        slc = slice(0, 10)
        self.assertEqual(slice_count(slc), 10)

        slc = slice(0, 10, 1)
        self.assertEqual(slice_count(slc), 10)

        slc = slice(0, 10, 2)
        self.assertEqual(slice_count(slc), 5)

        slc = slice(0, 10, 3)
        self.assertEqual(slice_count(slc), 4)

    def test_slice_count(self):
        """
        slice_count() tested agains an actual list.
        """
        lst = range(100)
        for i in range(9):
            for j in range(i,10):
                for k in range(1, 10):
                    slc = slice(i,j,k)
                    assert len(lst[slc]) == slice_count(slc)

if __name__ == '__main__':
    unittest.main()
