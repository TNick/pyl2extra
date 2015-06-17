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
import Image
import numpy
import os
import shutil
import tempfile
import unittest
from collections import OrderedDict


from pyl2extra.datasets.images import Images


def create_images(path, factor):
    """
    Creates a number of images for testing using various encodings.

    The result is a dictionary with keys being file names and values
    being  tuples of (category, image).

    TODO: this is duplicated code from
    `pyl2extra/datasets/img_dataset/tests/test_data_providers.py`
    """
    result = OrderedDict()

    for i in range(factor):
        img_file = os.path.join(path, 'rgba_file_%d.png' % i)
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        im.save(img_file)
        result[img_file] = ('rgba', im, 0)

        img_file = os.path.join(path, 'rgb_file_%d.png' % i)
        imarray = numpy.random.rand(100,50,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        im.save(img_file)
        result[img_file] = ('rgb', im, 1)

        img_file = os.path.join(path, 'greyscale_file_%d.png' % i)
        imarray = numpy.random.rand(50,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('L')
        im.save(img_file)
        result[img_file] = ('l', im, 2)

        img_file = os.path.join(path, 'black_white_file_%d.png' % i)
        imarray = numpy.random.rand(100,10,3) * 2
        im = Image.fromarray(imarray.astype('uint8')).convert('1')
        im.save(img_file)
        result[img_file] = ('bw', im, 3)

        img_file = os.path.join(path, 'rpalette_file_%d.png' % i)
        imarray = numpy.random.rand(10,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('P')
        im.save(img_file)
        result[img_file] = ('palette', im, 4)

        img_file = os.path.join(path, 'cmyk_file_%d.jpg' % i)
        imarray = numpy.random.rand(255,254,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('CMYK')
        im.save(img_file)
        result[img_file] = ('cmyk', im, 5)

        img_file = os.path.join(path, 'integer_file_%d.png' % i)
        imarray = numpy.random.rand(10,11,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('I')
        im.save(img_file)
        result[img_file] = ('integer', im, 6)

        img_file = os.path.join(path, 'float_file_%d.tif' % i)
        imarray = numpy.random.rand(999,999,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('F')
        im.save(img_file)
        result[img_file] = ('float', im, 7)

    return result


class TestGeneric(unittest.TestCase):
    """
    Tests for Images.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = None
        self.tmp_dir = tempfile.mkdtemp()
        self.testset = create_images(self.tmp_dir, 10)
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
