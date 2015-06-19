"""
Tests for clone_layer module.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import shutil
import tempfile
import unittest

from pyl2extra.models.clone_layer import Clone


class TestClone(unittest.TestCase):
    """
    Tests for Clone.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = Clone()
        self.tmp_dir = tempfile.mkdtemp()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.testee

    def test_csv_files_categ(self):
        """
        Test a csv file that has both categories and files.
        """
        pass

if __name__ == '__main__':
    if True:
        unittest.main()
    else:
        unittest.main(argv=['--verbose', 'TestGeneric'])
