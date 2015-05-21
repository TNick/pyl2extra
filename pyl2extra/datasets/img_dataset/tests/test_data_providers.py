"""
Tests for data providers.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import Image
import numpy
import os
import shutil
import tempfile
import unittest

from pyl2extra.datasets.img_dataset.data_providers import (DictProvider,
                                                           CsvProvider,
                                                           RandomProvider)

class TestDictProvider(unittest.TestCase):
    """
    Tests for DictProvider.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.keys = ['a', 'b', 'c', 'd']
        self.vlist = ['1', '2', '3', '4']
        self.data = dict(zip(self.keys, self.vlist))
        self.testee = DictProvider(self.data)

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.keys
        del self.vlist
        del self.data
        del self.testee

    def test_iteration(self):
        """
        Check the iteration for DictProvider
        """
        klist = []
        vlist = []
        for k in self.testee:
            klist.append(k)
            vlist.append(self.testee.category(k))
        klist.sort()
        vlist.sort()
        self.assertEqual(klist, self.keys)
        self.assertEqual(vlist, self.vlist)
        evrt = self.testee.everything().keys()
        evrt.sort()
        self.assertEqual(klist, evrt)

    def test_categ_len(self):
        """
        Check the categ_len() for DictProvider
        """
        self.assertEqual(self.testee.categ_len(), len(self.vlist))

    def test_categories(self):
        """
        Check the categories() for DictProvider
        """
        ctgs = self.testee.categories()
        ctgs.sort()
        self.assertEqual(ctgs, self.vlist)

    def test_categ2int(self):
        """
        Check the categ2int() for DictProvider
        """
        result = []
        for val in self.vlist:
            result.append(self.testee.categ2int(val))
        result.sort()
        self.assertEqual(result, [0, 1, 2, 3])

    def test_int2categ(self):
        """
        Check the int2categ() for DictProvider
        """
        result = []
        for val in [0, 1, 2, 3]:
            result.append(self.testee.int2categ(val))
        result.sort()
        self.assertEqual(result, self.vlist)

    def test_cnext(self):
        """
        Check the cnext() for DictProvider

        It should never trigger StopIteration.
        """
        for i in range(1000):
            self.assertTrue(self.testee.cnext() in self.keys)

    def test_get(self):
        """
        Get based on offset and count
        """
        result = self.testee.get(0, 4)
        result.sort()
        self.assertListEqual(result, self.keys)
        for i in range(0, 100):
            result = self.testee.get(i, 3)
            for relem in result:
                self.assertIn(relem, self.keys)
        
class TestDummyImages(unittest.TestCase):
    """
    Tests for DictProvider.
    
    lab not supported
    hsv not supported    
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.data = {}

        img_file = os.path.join(self.tmp_dir, 'rgba_file.png')
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        im.save(img_file)
        self.data[img_file] = 'rgba'

        img_file = os.path.join(self.tmp_dir, 'rgb_file.png')
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        im.save(img_file)
        self.data[img_file] = 'rgb'

        img_file = os.path.join(self.tmp_dir, 'greyscale_file.png')
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('L')
        im.save(img_file)
        self.data[img_file] = 'l'

        img_file = os.path.join(self.tmp_dir, 'black_white_file.png')
        imarray = numpy.random.rand(100,100,3) * 2
        im = Image.fromarray(imarray.astype('uint8')).convert('1')
        im.save(img_file)
        self.data[img_file] = 'bw'

        img_file = os.path.join(self.tmp_dir, 'rpalette_file.png')
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('P')
        im.save(img_file)
        self.data[img_file] = 'palette'

        img_file = os.path.join(self.tmp_dir, 'cmyk_file.jpg')
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('CMYK')
        im.save(img_file)
        self.data[img_file] = 'cmyk'

        img_file = os.path.join(self.tmp_dir, 'integer_file.png')
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('I')
        im.save(img_file)
        self.data[img_file] = 'integer'

        img_file = os.path.join(self.tmp_dir, 'float_file.tif')
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('F')
        im.save(img_file)
        self.data[img_file] = 'float'

        self.testee = DictProvider(self.data)

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.data
        del self.testee

    def test_read_file(self):
        """
        Check read() method
        """
        for fpath in self.testee.everything():
            imarray, categ = self.testee.read(fpath)
            self.assertEqual(len(imarray.shape), 3)
            self.assertEqual(imarray.shape[2], 4)
            self.assertTrue(fpath in self.data.keys())
            self.assertTrue(categ in self.data.values())
            self.assertTrue(str(imarray.dtype).startswith('float'))


class BaseCsvProvider():
    """
    Common code for tests for CsvProvider.
    """
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.testee

    def test_iteration(self):
        """
        Check the iteration for CsvProvider
        """
        klist = []
        vlist = []
#        try:
#            while True:
#                k = self.testee.next()
#                klist.append(k)
#                vlist.append(self.testee.category(k))            
#        except StopIteration:
#            pass
        for k in self.testee:
            klist.append(k)
            vlist.append(self.testee.category(k))
        klist.sort()
        vlist.sort()
        self.assertEqual(klist, self.keys)

        evrt = self.testee.everything().keys()
        evrt.sort()
        self.assertEqual(klist, evrt)

        vlist = [int(v) for v in vlist]
        self.assertEqual(vlist, self.vlist)

    def prepare(self):
        self.tmp_dir = tempfile.mkdtemp()
        keys = ['a', 'b', 'c', 'd']
        self.keys = []
        self.vlist = [1, 2, 3, 4]
        for key in keys:
            self.keys.append(os.path.join(self.tmp_dir, key))
        csv_file = os.path.join(self.tmp_dir, 'test.csv')
        return csv_file


class TestCsvProviderDefaultArangement(unittest.TestCase, BaseCsvProvider):
    """
    Tests for CsvProvider.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):

        csv_file = self.prepare()
        with open(csv_file, 'wt') as fhand:
            for key,value in zip(self.keys, self.vlist):
                fhand.write('%d,%s\n' % (value, key))

        self.testee = CsvProvider(csv_file)


class TestCsvProviderSwitchedArangement(unittest.TestCase, BaseCsvProvider):
    """
    Tests for CsvProvider.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):

        csv_file = self.prepare()
        with open(csv_file, 'wt') as fhand:
            for key,value in zip(self.keys, self.vlist):
                fhand.write('%s,%d\n' % (key, value))

        self.testee = CsvProvider(csv_file, col_path=0, col_class=1)


class TestCsvProviderSeparator(unittest.TestCase, BaseCsvProvider):
    """
    Tests for CsvProvider.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):

        csv_file = self.prepare()
        with open(csv_file, 'wt') as fhand:
            for key,value in zip(self.keys, self.vlist):
                fhand.write('%s>%d\n' % (key, value))

        self.testee = CsvProvider(csv_file, col_path=0,
                                  col_class=1, delimiter='>')


class TestCsvProviderQuote(unittest.TestCase, BaseCsvProvider):
    """
    Tests for CsvProvider.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):

        csv_file = self.prepare()
        with open(csv_file, 'wt') as fhand:
            for key,value in zip(self.keys, self.vlist):
                fhand.write('$%s$>$%d$\n' % (key, value))

        self.testee = CsvProvider(csv_file, col_path=0,
                                  col_class=1, delimiter='>',
                                  quotechar='$')


class TestCsvProviderMulCol(unittest.TestCase, BaseCsvProvider):
    """
    Tests for CsvProvider.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):

        csv_file = self.prepare()
        with open(csv_file, 'wt') as fhand:
            for key,value in zip(self.keys, self.vlist):
                fhand.write('a>"b">c>d>%s>x>y>z>t>"%d"\n' % (key, value))

        self.testee = CsvProvider(csv_file, col_path=4,
                                  col_class=9, delimiter='>')


class TestCsvProviderHeader(unittest.TestCase, BaseCsvProvider):
    """
    Tests for CsvProvider.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):

        csv_file = self.prepare()
        with open(csv_file, 'wt') as fhand:
            col_path = 'Path Column'
            col_class = 'Class Column'
            fhand.write('>"some">column>here>%s>thet>we\'re>'
                        'not>interested>"%s">in\n' % (col_path, col_class))
            for key,value in zip(self.keys, self.vlist):
                fhand.write('a>"b">c>d>%s>x>y>z>t>"%d"\n' % (key, value))

        self.testee = CsvProvider(csv_file, 
                                  has_header=True,
                                  col_path=col_path,
                                  col_class=col_class,
                                  delimiter='>')

class TestRandomProvider(unittest.TestCase):
    """
    Tests for RandomProvider.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = RandomProvider()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee

    def test_cnext(self):
        """
        Check the cnext() for DictProvider

        It should never trigger StopIteration.
        """
        for i in range(1000):
            imgfile = self.testee.cnext()
            self.assertEqual(imgfile, self.testee.category(imgfile))

    def test_read_file(self):
        """
        Check read() method
        """
        for fpath in self.testee.everything():
            imarray, categ = self.testee.read(fpath)
            self.assertEqual(len(imarray.shape), 3)
            self.assertEqual(imarray.shape[2], 4)
            self.assertEqual(fpath, categ)
            self.assertTrue(str(imarray.dtype).startswith('float'))

    def test_hash(self):
        """
        Check hash() method
        """
        self.assertTrue(hash(self.testee))
        
    def test_len(self):
        """
        Check len() method
        """
        self.assertTrue(len(self.testee) > 0)
        
if __name__ == '__main__':
    unittest.main()
