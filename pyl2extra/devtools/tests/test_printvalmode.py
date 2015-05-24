"""
Tests for dataset
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import logging
import numpy
import os
import shutil
import tempfile
import unittest

os.environ['THEANO_FLAGS'] = 'mode=DEBUG_MODE,' \
                             'exception_verbosity=high,' \
                             'optimizer=None'

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.tests.test_sgd import SupervisedDummyCost
from pylearn2.training_algorithms.tests.test_sgd import SoftmaxModel

from pyl2extra.devtools.printvalmode import PrintValMode, XmlValMode

class TestPrintValMode(unittest.TestCase):
    """
    Tests for ImgDataset
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        dim = 3
        m = 10

        rng = numpy.random.RandomState([25, 9, 2012])

        X = rng.randn(m, dim)

        idx = rng.randint(0, dim, (m, ))
        Y = numpy.zeros((m, dim))
        for i in xrange(m):
            Y[i, idx[i]] = 1

        dataset = DenseDesignMatrix(X=X, y=Y)

        m = 15
        X = rng.randn(m, dim)

        idx = rng.randint(0, dim, (m,))
        Y = numpy.zeros((m, dim))
        for i in xrange(m):
            Y[i, idx[i]] = 1

        monitoring_dataset = DenseDesignMatrix(X=X, y=Y)
        model = SoftmaxModel(dim)
        learning_rate = 1e-3
        batch_size = 5
        cost = SupervisedDummyCost()
        termination_criterion = EpochCounter(1)

        self.algorithm = SGD(learning_rate, cost,
                             batch_size=batch_size,
                             monitoring_batches=3,
                             monitoring_dataset=monitoring_dataset,
                             termination_criterion=termination_criterion,
                             update_callbacks=None,
                             set_batch_size=False,
                             theano_function_mode=None)

        self.train = Train(dataset,
                           model,
                           self.algorithm,
                           save_path=None,
                           save_freq=0,
                           extensions=None)


    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_print(self):
        """
        """
        self.algorithm.theano_function_mode = PrintValMode()

        logging.debug('Entering train main loop')
        self.train.main_loop(time_budget=100)
        logging.debug('Train main loop done')

    def test_xml(self):
        """
        """
        out_f = os.path.join(self.tmp_dir, 'out.xml')
        with open(out_f, 'wt') as fhand:
            self.algorithm.theano_function_mode = XmlValMode(fhand)

            logging.debug('Entering train main loop')
            self.train.main_loop(time_budget=100)
            logging.debug('Train main loop done')

            self.algorithm.theano_function_mode.tear_down()
        print out_f
        self.assertTrue(os.path.isfile(out_f))

if __name__ == '__main__':
    if False:
        unittest.main(argv=['--verbose', 'TestPrintValMode.test_xml'])
    else:
        unittest.main(argv=['--verbose'])
