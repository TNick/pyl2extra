# -*- coding: utf-8 -*-
"""
Costs helpers.

:class:`SupervisedDummyCost` and :class:`DummyCost` were copied
from ``pylearn2/training_algorithms/tests/test_sgd.py`` because
``test_sgd`` can't be imported.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ['PyLearn2 team']
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
import theano.tensor as T


class SupervisedDummyCost(DefaultDataSpecsMixin, Cost):
    """
    A dummy cost copied from test_sgd.
    """
    supervised = True

    def expr(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        return T.square(model(X) - Y).mean()


class DummyCost(DefaultDataSpecsMixin, Cost):
    """
    A dummy cost copied from test_sgd.
    """
    def expr(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X = data
        return T.square(model(X) - X).mean()
