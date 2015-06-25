#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for overfeat module.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import numpy
import os
import shutil
import tempfile
import unittest

os.environ['THEANO_FLAGS'] = 'optimizer=fast_compile,' \
    'device=cpu,' \
    'floatX=float64'
#    'linker=py,'
#    'exception_verbosity=high,' \

from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.costs.cost import MethodCost

from pyl2extra.testing import images
from pyl2extra.models.overfeat import (Params, standardize, FILE_PATH_KEY,
    SMALL_NETWORK_FILTER_SHAPES, SMALL_NETWORK_BIAS_SHAPES,
    LARGE_NETWORK_FILTER_SHAPES, LARGE_NETWORK_BIAS_SHAPES,
    SMALL_INPUT, LARGE_INPUT, predict, StandardizePrep)



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


class ModelBaseTst(unittest.TestCase):
    """
    Base class for both large and small.
    """
    def prepare(self, large, inp_size):
        self.testee = None
        self.tmp_dir = tempfile.mkdtemp()
        self.images = images.create(self.tmp_dir)
        self.dataset = images.dataset(
            image_size=inp_size,
            path=self.tmp_dir,
            factor=10,
            preprocessor=StandardizePrep())
        self.valid_dataset = images.dataset(
            inp_size,
            self.tmp_dir,
            factor=10,
            preprocessor=StandardizePrep())
        self.testee = Params(large=large, weights_file=None)
        self.train_obj = Train(
            dataset=self.dataset,
            model=self.testee.model(),
            algorithm=SGD(
                learning_rate=0.01,
                batch_size=2,
                learning_rule=Momentum(
                    init_momentum=0.5
                ),
                cost=MethodCost(
                    method='cost_from_X'
                ),
                termination_criterion=EpochCounter(
                    max_epochs=2
                ),
                monitoring_dataset={
                    'train': self.dataset
                }
            ),
            save_freq=0
        )

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.testee


class TestModelLarge(ModelBaseTst):
    """
    Tests for large model.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.prepare(True, LARGE_INPUT)

    def test_main_loop(self):
        """
        Loading the parameters for network.
        """
        self.train_obj.main_loop()

    def test_predict(self):
        """
        Predict using this network.
        """
        predict(model=self.mode, images=(self.images,))


class TestModelSmall(ModelBaseTst):
    """
    Tests small model.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.prepare(False, SMALL_INPUT)

    def test_main_loop(self):
        """
        Loading the parameters for network.
        """
        self.train_obj.main_loop()

    def test_predict(self):
        """
        Predict using this network.
        """
        predict(model=self.mode, images=(self.images,))


class TestPredict(ModelBaseTst):
    """
    Tests small model.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.images = images.create(self.tmp_dir)
        self.dataset = images.dataset(
            image_size=LARGE_INPUT,
            path=self.tmp_dir,
            factor=1,
            preprocessor=StandardizePrep())

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.images
        del self.dataset

    def test_predict_dataset(self):
        """
        Predict using a new network.
        """
        probabilities, classes, class_names = predict(images=self.dataset)
        self.assertTupleEqual(probabilities.shape, (8,8))
        self.assertEqual(len(classes), 8)
        self.assertEqual(len(class_names), 8)
        self.assertTrue(all(classes < 8))
        self.assertTrue(all(classes >= 0))

    def test_predict_image(self):
        """
        Predict using a new network.
        """
        probabilities, classes, class_names = predict(images=self.images[0])


if __name__ == '__main__':
    if False:
        unittest.main()
    else:
        unittest.main(argv=['--verbose', 'TestPredict.test_predict_dataset'])
        #tr = TestModelLarge()
        #tr.setUp()
        #tr.test_file()
        #tr.tearDown()
