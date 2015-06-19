"""
Tests for images module.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import csv
import functools
import os
import shutil
import tempfile
import unittest

from pyl2extra.datasets.images import Images
from pyl2extra.testing import images


class TestGeneric(unittest.TestCase):
    """
    Tests for Images.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = None
        self.tmp_dir = tempfile.mkdtemp()
        self.testset = images.create(self.tmp_dir, 10)
        self.csv_file_fc = os.path.join(self.tmp_dir, 'files_and_categs.csv')
        with open(self.csv_file_fc, 'wt') as fhand:
            csvw = csv.writer(fhand, delimiter=',', quotechar='"')
            for fpath in self.testset:
                csvw.writerow([self.testset[fpath][2], fpath])
        self.csv_file_f = os.path.join(self.tmp_dir, 'files.csv')
        with open(self.csv_file_f, 'wt') as fhand:
            csvw = csv.writer(fhand, delimiter=',', quotechar='"')
            for fpath in self.testset:
                csvw.writerow(['', fpath])

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.testee
        del self.testset

    def test_csv_files_categ(self):
        """
        Test a csv file that has both categories and files.
        """
        self.testee = Images(source=self.csv_file_fc,
                             image_size=None, regression=False)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertTupleEqual(self.testee.y.shape, (len(self.testset), 1))
        self.assertTrue(self.testee.has_targets())

    def test_csv_files(self):
        """
        Test a csv file that has only files.
        """
        self.testee = Images(source=self.csv_file_f,
                             image_size=None, regression=False)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertIsNone(self.testee.y)
        self.assertFalse(self.testee.has_targets())

    def test_dict_categs(self):
        """
        Test with a dictionary of file paths and categories.
        """
        tst = {}
        for fpath in self.testset:
            tst[fpath] = self.testset[fpath][2]
        self.testee = Images(source=tst,
                             image_size=79, regression=False)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertTupleEqual(self.testee.y.shape, (len(self.testset), 1))
        self.assertTrue(self.testee.has_targets())

    def test_dict_no_categs(self):
        """
        Test with a dictionary of file paths and no categories.
        """
        tst = {}
        for fpath in self.testset:
            tst[fpath] = None
        self.testee = Images(source=tst,
                             image_size=79, regression=False)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertIsNone(self.testee.y)
        self.assertFalse(self.testee.has_targets())

    def test_dict_images(self):
        """
        Test with a dictionary of images and categories.
        """
        tst = {}
        for fpath in self.testset:
            tst[self.testset[fpath][1]] = self.testset[fpath][2]
        self.testee = Images(source=tst,
                             image_size=79, regression=False)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertTupleEqual(self.testee.y.shape, (len(self.testset), 1))
        self.assertTrue(self.testee.has_targets())

    def test_list_images(self):
        """
        Test with a list of image paths and no categories.
        """
        lst_img = []
        for fpath in self.testset:
            lst_img.append(fpath)
        self.testee = Images(source=[lst_img],
                             image_size=79, regression=False)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertIsNone(self.testee.y)
        self.assertFalse(self.testee.has_targets())

    def test_list_images_and_categs(self):
        """
        Test with a list of image paths and categories.
        """
        lst_img = []
        lst_categ = []
        for fpath in self.testset:
            lst_img.append(fpath)
            lst_categ.append(self.testset[fpath][2])
        self.testee = Images(source=(lst_img, lst_categ),
                             image_size=79, regression=False)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertTupleEqual(self.testee.y.shape, (len(self.testset), 1))
        self.assertTrue(self.testee.has_targets())

if __name__ == '__main__':
    if True:
        unittest.main()
    else:
        unittest.main(argv=['--verbose', 'TestGeneric'])
