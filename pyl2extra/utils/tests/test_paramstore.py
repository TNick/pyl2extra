"""
Tests for paramstore..
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import unittest

from pyl2extra.utils.paramstore import ParamStore


class TestParamStore(unittest.TestCase):
    """
    Tests for slice_count().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.colors = ['white', 'red', 'green', 'blue', 'black']
        self.numbers = [1, 2, 3]
        self.testee = ParamStore(parameters=[self.colors, self.numbers],
                                 mode="rand_one",
                                 rng=[2017, 7, 7])

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.colors
        del self.numbers
        del self.testee

    def test_seq_all(self):
        """
        Check next_seq_all()
        """
        self.testee.reset('seq_all')

        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('white', 1))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('red', 2))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('green', 3))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('blue', 1))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('black', 2))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('white', 3))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('red', 1))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('green', 2))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('blue', 3))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('black', 1))

    def test_seq_one(self):
        """
        Check next_seq_one()
        """
        self.testee.reset('seq_one')
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('white', 1))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('white', 2))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('red', 2))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('red', 3))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('green', 3))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('green', 1))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('blue', 1))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('blue', 2))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('black', 2))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('black', 3))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('white', 3))
        rslt = self.testee.next()
        self.assertTupleEqual(rslt, ('white', 1))

    def test_rand_all(self):
        """
        Check next_rand_all()
        """
        self.testee.reset('rand_all')
        result = {}
        for i in range(1000):
            rslt = self.testee.next()
            if rslt[0] in result:
                if not rslt[1] in result[rslt[0]]:
                    result[rslt[0]].append(rslt[1])
            else:
                result[rslt[0]] = [rslt[1]]
        self.colors.sort()
        keys = result.keys()
        keys.sort()
        self.assertListEqual(self.colors, keys)

        for k in keys:
            vallist = result[k]
            vallist.sort()
            self.assertListEqual(self.numbers, vallist)

    def test_rand_one(self):
        """
        Check next_rand_one()
        """
        self.testee.reset('rand_one')
        result = {}
        for i in range(1000):
            rslt = self.testee.next()
            if rslt[0] in result:
                if not rslt[1] in result[rslt[0]]:
                    result[rslt[0]].append(rslt[1])
            else:
                result[rslt[0]] = [rslt[1]]
        self.colors.sort()
        keys = result.keys()
        keys.sort()
        self.assertListEqual(self.colors, keys)

        for k in keys:
            vallist = result[k]
            vallist.sort()
            self.assertListEqual(self.numbers, vallist)


if __name__ == '__main__':
    unittest.main()
