"""
Tests for dataset
"""

import functools
import Image
import logging
import numpy
import os
import shutil
import tempfile
import unittest
import pickle
from pylearn2.space import CompositeSpace, VectorSpace, Conv2DSpace, IndexSpace


from pyl2extra.datasets.img_dataset.data_providers import RandomProvider
from pyl2extra.datasets.img_dataset.generators import (InlineGen, Generator,
                                                       ProcessGen,
                                                       ThreadedGen,
                                                       gen_from_string)
from pyl2extra.datasets.img_dataset.adjusters import (FlipAdj, RotationAdj,
                                                      ScalePatchAdj, GcaAdj,
                                                      MakeSquareAdj, Adjuster,
                                                      BackgroundAdj,
                                                      adj_from_string)
from pyl2extra.datasets.img_dataset.dataset import ImgDataset

class BaseImgDset(object):
    """
    """
    def __init__(self):
        super(BaseImgDset, self).init()

    def prepare(self):
        self.tmp_dir = tempfile.mkdtemp()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        self.testee.tear_down()
        del self.testee

    def test_basic_iterator(self):
        """
        Test the iterator with default values.
        """
        itr = self.testee.iterator(mode=None,
                                   batch_size=53,
                                   num_batches=None,
                                   rng=None,
                                   data_specs=None,
                                   return_tuple=False)
        r = itr.next()
        self.assertEqual(len(r), 2)
        self.assertEqual(len(r[0].shape), 4)
        self.assertEqual(r[0].shape[0], 53)
        self.assertEqual(r[0].shape[1], 128)
        self.assertEqual(r[0].shape[2], 128)
        self.assertEqual(r[0].shape[3], 3)

        self.assertEqual(len(r[1].shape), 2)
        self.assertEqual(r[1].shape[0], 53)
        self.assertEqual(r[1].shape[1], 1)


class TestImgDatasetConstructor(unittest.TestCase, BaseImgDset):
    """
    Tests for ImgDataset
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.prepare()
        data_provider = RandomProvider()
        generator = InlineGen()
        adjusters = None
        self.testee = ImgDataset(data_provider=data_provider,
                                 adjusters=adjusters,
                                 generator=generator,
                                 shape=None,
                                 axes=None,
                                 cache_loc=self.tmp_dir,
                                 rng=None)

    def test_constructor(self):
        """
        Make sure the constructor does its job.
        """
        self.assertEqual(self.testee.shape, (128, 128))
        self.assertEqual(self.testee.axes, ('b', 0, 1, 'c'))
        self.assertIsInstance(self.testee.data_provider, RandomProvider)
        for adj in self.testee.adjusters:
            self.assertIsInstance(adj, Adjuster)
        self.assertIsInstance(self.testee.generator, Generator)
        self.assertEqual(self.testee.cache_loc, self.tmp_dir)
        self.assertNotEqual(self.testee.hash_value, 0)
        self.assertIs(self.testee.first_batch, None)
        self.assertEqual(self.testee.totlen, 100)

        self.assertIsInstance(self.testee.rng, numpy.random.RandomState)

        self.assertEqual(self.testee._iter_batch_size, 128)
        self.assertGreater(self.testee._iter_num_batches, 0)
        (space, source) = self.testee._data_specs()
        self.assertIsInstance(space, CompositeSpace)
        self.assertIsInstance(space.components[0], Conv2DSpace)
        self.assertEqual(space.components[0].axes, ('b', 0, 1, 'c'))
        self.assertEqual(space.components[0].num_channels, 3)
        self.assertIsInstance(space.components[1], IndexSpace)
        self.assertEqual(space.components[1].max_labels, 100)
        self.assertEqual(space.components[1].dim, 1)
        self.assertEqual(len(source), 2)
        self.assertEqual(source[0], 'features')
        self.assertEqual(source[1], 'targets')


        self.assertTrue(self.testee.has_targets)
        self.assertEqual(self.testee.get_num_examples(), self.testee.totlen)
        self.assertEqual(len(self.testee), self.testee.totlen)
        self.assertEqual(self.testee.get_data_specs(),
                         self.testee._data_specs())

        self.assertFalse(self.testee._iter_subset_class is None)

    #def test_iterator(self):
    #    self.test_basic_iterator()


class TestImgDatasetBackground(unittest.TestCase, BaseImgDset):
    """
    Tests for ImgDataset
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.prepare()
        data_provider = RandomProvider()
        generator = InlineGen()
        adjusters = [BackgroundAdj()]
        self.testee = ImgDataset(data_provider=data_provider,
                                 adjusters=adjusters,
                                 generator=generator,
                                 shape=None,
                                 axes=None,
                                 cache_loc=self.tmp_dir,
                                 rng=None)

    def test_background(self):
        """
        Test the background adjuster.
        """
        pass


def create_test_img_dataset(cache_loc, gen='inline'):
        data_provider = RandomProvider()
        generator = gen_from_string(gen)
        adjusters = [MakeSquareAdj(size=128),
                     BackgroundAdj(backgrounds=[(0, 0, 64),
                                                (0, 64, 64),
                                                (64, 64, 64),
                                                (0, 0, 128),
                                                (0, 128, 128),
                                                (128, 128, 128),
                                                (0, 0, 255),
                                                (0, 255, 255),
                                                (255, 255, 255),
                                                (0, 0, 0)]),
                     FlipAdj(horizontal=True,
                             vertical=False),
                     RotationAdj(min_deg=-45.0,
                                 max_deg=45.0,
                                 step=15.0),
                     ScalePatchAdj(outsize=None,
                                   start_factor=0.8,
                                   end_factor=0.9,
                                   step=0.1,
                                   placements=None),
                     GcaAdj(start_scale=1.,
                            end_scale=1.,
                            step_scale=0.,
                            subtract_mean=None, use_std=None,
                            start_sqrt_bias=0.,
                            end_sqrt_bias=0.,
                            step_sqrt_bias=0.)]
        return ImgDataset(data_provider=data_provider,
                          adjusters=adjusters,
                          generator=generator,
                          shape=None,
                          axes=None,
                          cache_loc=cache_loc,
                          rng=None)

class TestImgDatasetAll(unittest.TestCase, BaseImgDset):
    """
    Tests for ImgDataset
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.prepare()
        self.testee = create_test_img_dataset(self.tmp_dir)

    def test_process(self):
        """
        Test the background adjuster.
        """
        itr = self.testee.iterator(mode=None,
                                   batch_size=None,
                                   num_batches=None,
                                   rng=None,
                                   data_specs=None,
                                   return_tuple=False)
        result = itr.next()
        #print result

class TestImgDatasetPickle(unittest.TestCase):
    """
    Tests for ImgDataset
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.testee = create_test_img_dataset(self.tmp_dir)
        
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee
        shutil.rmtree(self.tmp_dir)
    
    def test_pickle(self):
        """
        Make sure that we can pickle the dataset.
        """
        pkl = pickle.dumps(self.testee)
        reload_tt = pickle.loads(pkl)
        self.assertEqual(len(self.testee.adjusters), len(reload_tt.adjusters)) 


    

#def explore_pick():
#    testee = create_test_img_dataset("/var/tmp/xxxx")
#    pkl = pickle.dumps(testee)
#    reload_tt = pickle.loads(pkl)
    #for de in testee.__dict__:
    #    print de, de.__class__

#    for de in testee.data_provider.__dict__:
#        print de, testee.data_provider.__dict__[de].__class__
#    for adj in testee.adjusters:
#        for de in adj.__dict__:
#            print de, adj.__dict__[de].__class__
#    for de in testee.generator.__dict__:
#        print de, testee.generator.__dict__[de].__class__
        
#    for de in testee.__dict__:
#        obj = testee.__dict__[de]
#        print de, obj.__class__
#        pickle.dumps(obj)
#    print '-' * 64
#    for de in testee.data_provider.__dict__:
#        obj = testee.data_provider.__dict__[de]
#        print de, obj.__class__
#        if not de in ['dataset']:
#            pickle.dumps(obj)
#        print '*' * 64
    
def simple_stand_alone_test():
    root_logger = logging.getLogger()
    if 1:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)
    imd = create_test_img_dataset("/var/tmp/xxxx", gen='process')
    itr = imd.iterator(mode=None,
                       batch_size=4,
                       num_batches=4,
                       rng=None,
                       data_specs=None,
                       return_tuple=False)
    result = itr.next()
    imd.tear_down()
    print result

if __name__ == '__main__':
    simple_stand_alone_test()
    #explore_pick()
    #unittest.main()