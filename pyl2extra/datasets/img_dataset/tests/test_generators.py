"""
Tests for adjusters.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import unittest

from pyl2extra.datasets.img_dataset.generators import (InlineGen, 
                                                       ThreadedGen,
                                                       ProcessGen,
                                                       genFromString)


class TestInlineGen(unittest.TestCase):
    """
    Tests for InlineGen.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = InlineGen()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee
        
    def test_is_inline(self):
        """
        Check the transf_count() for InlineGen
        """
        instloc = InlineGen()
        self.assertTrue(instloc.is_inline())
        

class TestThreadedGen(unittest.TestCase):
    """
    Tests for ThreadedGen.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = ThreadedGen()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee
        
    def test_is_inline(self):
        """
        Check the transf_count() for ThreadedGen
        """
        instloc = ThreadedGen()
        self.assertFalse(instloc.is_inline())
        

class TestProcessGen(unittest.TestCase):
    """
    Tests for ProcessGen.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = ProcessGen()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee
        
    def test_is_inline(self):
        """
        Check the transf_count() for ProcessGen
        """
        instloc = ProcessGen()
        self.assertFalse(instloc.is_inline())
        
        
class TestGenFromString(unittest.TestCase):
    """
    Tests for genFromString().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        pass

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_simple(self):
        """
        Create generators via genFromString().
        """
        adj = genFromString('inline')
        self.assertIsInstance(adj, InlineGen)
        adj = genFromString('threads')
        self.assertIsInstance(adj, ThreadedGen)
        adj = genFromString('process')
        self.assertIsInstance(adj, ProcessGen)


if __name__ == '__main__':
    unittest.main()
