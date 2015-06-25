# -*- coding: utf-8 -*-
"""
Costs helpers.

:class:`DummyModel`, :class:`SoftmaxModel` and :class:`TopoSoftmaxModel`
were copied from
``pylearn2/training_algorithms/tests/test_sgd.py`` because
``test_sgd`` can't be imported.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ['PyLearn2 team']
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import numpy as np
from pylearn2.models.model import Model
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.utils import sharedX
import theano.tensor as T


class DummyModel(Model):
    """
    A dummy model used for testing.

    Parameters
    ----------
    shapes : list
        List of shapes for each parameter.
    lr_scalers : list, optional
        Scalers to use for each parameter.
    init_type : string, optional
        How to fill initial values in parameters: `random` - generate random
        values; `zeros` - set all to zeros.
    """
    def __init__(self, shapes, lr_scalers=None, init_type='random'):
        super(DummyModel, self).__init__()
        if init_type == 'random':
            self._params = [sharedX(np.random.random(shp)) for shp in shapes]
        elif init_type == 'zeros':
            self._params = [sharedX(np.zeros(shp)) for shp in shapes]
        else:
            raise ValueError('Unknown value for init_type: %s',
                             init_type)
        self.input_space = VectorSpace(1)
        self.lr_scalers = lr_scalers

    def __call__(self, X):
        # Implemented only so that DummyCost would work
        return X

    def get_lr_scalers(self):
        if self.lr_scalers:
            return dict(zip(self._params, self.lr_scalers))
        else:
            return dict()


class SoftmaxModel(Model):
    """A dummy model used for testing.
       Important properties:
           has a parameter (P) for SGD to act on
           has a get_output_space method, so it can tell the
           algorithm what kind of space the targets for supervised
           learning live in
           has a get_input_space method, so it can tell the
           algorithm what kind of space the features live in
    """

    def __init__(self, dim):
        super(SoftmaxModel, self).__init__()
        self.dim = dim
        rng = np.random.RandomState([2012, 9, 25])
        self.P = sharedX(rng.uniform(-1., 1., (dim, )))

    def get_params(self):
        return [self.P]

    def get_input_space(self):
        return VectorSpace(self.dim)

    def get_output_space(self):
        return VectorSpace(self.dim)

    def __call__(self, X):
        # Make the test fail if algorithm does not
        # respect get_input_space
        assert X.ndim == 2
        # Multiplying by P ensures the shape as well
        # as ndim is correct
        return T.nnet.softmax(X*self.P)


class TopoSoftmaxModel(Model):
    """A dummy model used for testing.
       Like SoftmaxModel but its features have 2 topological
       dimensions. This tests that the training algorithm
       will provide topological data correctly.
    """

    def __init__(self, rows, cols, channels):
        super(TopoSoftmaxModel, self).__init__()
        dim = rows * cols * channels
        self.input_space = Conv2DSpace((rows, cols), channels)
        self.dim = dim
        rng = np.random.RandomState([2012, 9, 25])
        self.P = sharedX(rng.uniform(-1., 1., (dim, )))

    def get_params(self):
        return [self.P]

    def get_output_space(self):
        return VectorSpace(self.dim)

    def __call__(self, X):
        # Make the test fail if algorithm does not
        # respect get_input_space
        assert X.ndim == 4
        # Multiplying by P ensures the shape as well
        # as ndim is correct
        return T.nnet.softmax(X.reshape((X.shape[0], self.dim)) * self.P)
