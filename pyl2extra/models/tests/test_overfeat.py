"""
Tests for overfeat module.
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
from itertools import izip
import numpy
import os
import shutil
import tempfile
import unittest

from pyl2extra.testing import images

from pyl2extra.models.overfeat import (Params, standardize, FILE_PATH_KEY,
    SMALL_NETWORK_FILTER_SHAPES, SMALL_NETWORK_BIAS_SHAPES,
    LARGE_NETWORK_FILTER_SHAPES, LARGE_NETWORK_BIAS_SHAPES,
    SMALL_INPUT, LARGE_INPUT)



@unittest.skipUnless(os.environ.has_key(FILE_PATH_KEY), 
                     'missing %s environment variable' % FILE_PATH_KEY)
class TestParams(unittest.TestCase):
    """
    Tests for Params.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = None
        self.weights_path = os.environ[FILE_PATH_KEY]
        self.tmp_dir = tempfile.mkdtemp()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.testee

    def test_small_no_file(self):
        """
        Loading the parameters for small network.
        """
        self.testee = Params(large=False)
        self.assertFalse(self.testee.large)
        self.assertTrue(os.path.isfile(self.testee.weights_file))
        self.assertIsInstance(self.testee.filter_shapes, numpy.ndarray)
        self.assertIsInstance(self.testee.biases_shapes, numpy.ndarray)
        self.assertIsInstance(self.testee.weights, numpy.ndarray)
        self.assertIsInstance(self.testee.biases, numpy.ndarray)
        
        self.assertTupleEqual(self.testee.filter_shapes.shape, (8, 4))
        self.assertTupleEqual(self.testee.biases_shapes.shape, (8,))
        self.assertTupleEqual(self.testee.weights.shape, (8,))
        self.assertTupleEqual(self.testee.biases.shape, (8,))
        
        for i, (weight, bias) in enumerate(zip(self.testee.weights,
                                             self.testee.biases)):
            self.assertTupleEqual(weight.shape,
                                  tuple(SMALL_NETWORK_FILTER_SHAPES[i]))
            self.assertTupleEqual(bias.shape,
                                  (int(SMALL_NETWORK_BIAS_SHAPES[i]), ))

    def test_large_no_file(self):
        """
        Loading the parameters for small network.
        """
        self.testee = Params(large=True)
        self.assertTrue(self.testee.large)
        self.assertTrue(os.path.isfile(self.testee.weights_file))
        self.assertIsInstance(self.testee.filter_shapes, numpy.ndarray)
        self.assertIsInstance(self.testee.biases_shapes, numpy.ndarray)
        self.assertIsInstance(self.testee.weights, numpy.ndarray)
        self.assertIsInstance(self.testee.biases, numpy.ndarray)
        
        self.assertTupleEqual(self.testee.filter_shapes.shape, (9, 4))
        self.assertTupleEqual(self.testee.biases_shapes.shape, (9,))
        self.assertTupleEqual(self.testee.weights.shape, (9,))
        self.assertTupleEqual(self.testee.biases.shape, (9,))
        
        for i, (weight, bias) in enumerate(zip(self.testee.weights,
                                             self.testee.biases)):
            self.assertTupleEqual(weight.shape,
                                  tuple(LARGE_NETWORK_FILTER_SHAPES[i]))
            self.assertTupleEqual(bias.shape,
                                  (int(LARGE_NETWORK_BIAS_SHAPES[i]), ))


class TestStandardize(unittest.TestCase):
    """
    Tests for standardize.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.images = images.create(self.tmp_dir)

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir

    def test_file(self):
        """
        Loading the parameters for small network.
        """
        for ipath in self.images:
            ary = standardize(ipath)
            self.assertEqual(ary.shape[2], 3)

from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.train import Train
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor
from pylearn2.training_algorithms.sgd import LinearDecayOverEpoch
from pylearn2.train_extensions.window_flip import WindowAndFlip
from pylearn2.costs.cost import MethodCost

class TestModelLarge(unittest.TestCase):
    """
    Tests for standardize.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.images = images.create(self.tmp_dir)
        self.dataset = images.dataset(LARGE_INPUT, self.tmp_dir)
        self.valid_dataset = images.dataset(LARGE_INPUT, self.tmp_dir)
        self.algorithm = SGD(
            batch_size=2,
            learning_rate=.17,
            learning_rule=Momentum(init_momentum=.5),
            monitoring_dataset={'test': self.valid_dataset},
            cost=MethodCost(method='cost_from_X'),
            termination_criterion=EpochCounter(max_epochs=2))
        self.train_obj = Train(
            dataset=self.dataset,
            model=None,
            algorithm=self.algorithm,
            save_path=None,
            save_freq=0)
         
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir

    def test_file(self):
        """
        Loading the parameters for small network.
        """
        self.testee = Params(large=True)
        model = self.testee.model()
        self.assertEqual(len(model.layers), 9)
        
        self.train_obj.model = model
        self.train_obj.main_loop()
            
            

if __name__ == '__main__':
    if True:
        unittest.main()
    else:
        unittest.main(argv=['--verbose', 'TestGeneric'])
