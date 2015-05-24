"""
Code for printing values as they are being computed by Theano.

This mode is inspired by nan_guard module in pylearn2.devtools.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import os
import shutil
import tempfile
import unittest

from pyl2extra.scripts.txml2html import Txml2Html



class TestTxml2Html(unittest.TestCase):
    """
    Tests for ImgDataset
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.base_name = os.path.join(self.tmp_dir, 'out')
        self.xml_file = os.path.split(os.path.abspath(__file__))[0]
        self.xml_file = os.path.join(self.xml_file, 'test_txml2html.xml')
        self.x2m = Txml2Html(self.xml_file, self.base_name)

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def file_check(self):
        """
        Check to make suire file exist
        """
        out_html = '%s.html' % self.base_name
        out_png = '%s.png' % self.base_name
        out_cmapx = '%s.cmapx' % self.base_name
        self.assertTrue(os.path.isfile(out_html))
        self.assertTrue(os.path.isfile(out_png))
        self.assertTrue(os.path.isfile(out_cmapx))

    def test_gen_full(self):
        """
        Tests gen_full function.
        """
        self.x2m.gen_full()
        self.file_check()

    def test_gen_inouts(self):
        """
        Tests gen_inouts function.
        """
        self.x2m.gen_inouts()
        self.x2m.parse()
        self.file_check()

    def test_gen_outputs(self):
        """
        Tests gen_outputs function.
        """
        self.x2m.gen_outputs()
        self.file_check()

    def test_gen_inputs(self):
        """
        Tests gen_inputs function.
        """
        self.x2m.gen_inputs()
        self.file_check()



if __name__ == '__main__':
    if False:
        unittest.main(argv=['--verbose', 'TestTxml2Html.test_gen_full'])
    else:
        unittest.main(argv=['--verbose'])
