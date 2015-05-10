"""
A module defining the CosDataset class.
"""

from pylearn2.datasets.dataset import Dataset
from pylearn2.utils import safe_izip, wraps
from pylearn2.utils.iteration import resolve_iterator_class, SubsetIterator
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils.rng import make_np_rng
import numpy as np
from theano import config

#'shuffled_sequential',
_banned_mods = [
                'random_slice',
                'random_uniform',
                'batchwise_shuffled_sequential',
                'even_sequential',
                'even_shuffled_sequential',
                'even_batchwise_shuffled_sequential',
                'even_sequences']


class RandomSubsetIterator(SubsetIterator):
    """
    Returns mini-batches proceeding randomly through the dataset.

    Notes
    -----
    Returns slice objects to represent ranges of indices (`fancy = False`).

    See :py:class:`SubsetIterator` for detailed constructor parameter
    and attribute documentation.
    """

    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        if rng is None:
            raise ValueError("None rng argument not supported for "
                             "random batch iteration")
        assert num_batches is None or num_batches >= 0
        self._dataset_size = dataset_size
        if batch_size is None:
            if num_batches is not None:
                try:
                    batch_size = int(np.ceil(self._dataset_size / num_batches))
                except OverflowError:
                    raise ValueError("Dataset size is too large. "
                                     "Please specify manually the "
                                     "size of the batch.")
            else:
                raise ValueError("need one of batch_size, num_batches "
                                 "for sequential batch iteration")
        elif batch_size is not None:
            if num_batches is not None:
                max_num_batches = np.ceil(self._dataset_size / batch_size)
                if num_batches > max_num_batches:
                    raise ValueError("dataset of %d examples can only provide "
                                     "%d batches with batch_size %d, but %d "
                                     "batches were requested" %
                                     (self._dataset_size, max_num_batches,
                                      batch_size, num_batches))
            else:
                num_batches = np.ceil(self._dataset_size / batch_size)
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._next_batch_no = 0
        self._idx = 0
        self._batch = 0

    @wraps(SubsetIterator.next, assigned=(), updated=())
    def next(self):
        if self._batch >= self.num_batches or self._idx >= self._dataset_size:
            raise StopIteration()

        # this fix the problem where dataset_size % batch_size != 0
        elif (self._idx + self._batch_size) > self._dataset_size:
            self._last = slice(self._idx, self._dataset_size)
            self._idx = self._dataset_size
            return self._last

        else:
            self._last = slice(self._idx, self._idx + self._batch_size)
            self._idx += self._batch_size
            self._batch += 1
            return self._last

    def __next__(self):
        return self.next()

    fancy = False
    stochastic = True
    uniform_batch_size = True

    @property
    @wraps(SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        product = self.batch_size * self.num_batches
        return min(product, self._dataset_size)

    @property
    @wraps(SubsetIterator.uneven, assigned=(), updated=())
    def uneven(self):
        return self.batch_size * self.num_batches > self._dataset_size

class InfiniteDatasetIterator(object):
    """
    A wrapper around subset iterators that actually retrieves
    data.
    """

    def __init__(self, dataset, subset_iterator, data_specs=None,
                 return_tuple=False, convert=None):
        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

        space, source = data_specs
        if not isinstance(source, tuple):
            source = (source,)
        if not isinstance(space, CompositeSpace):
            sub_spaces = (space,)
        else:
            sub_spaces = space.components
        assert len(source) == len(sub_spaces)

        # If `dataset` is incompatible with the new interface, fall back to the
        # old interface
        if not hasattr(self._dataset, 'get'):
            raise ValueError("InfiniteDatasetIterator supports only new "
                             "(get) interface for datasets.")

        self._source = source
        self._space = sub_spaces

        if convert is None:
            self._convert = [None for s in source]
        else:
            assert len(convert) == len(source)
            self._convert = convert

    def __iter__(self):
        return self

    @wraps(SubsetIterator.next)
    def next(self):
        """
        Retrieves the next batch of examples.

        Returns
        -------
        next_batch : object
            An object representing a mini-batch of data, conforming
            to the space specified in the `data_specs` constructor
            argument to this iterator. Will be a tuple if more
            than one data source was specified or if the constructor
            parameter `return_tuple` was `True`.

        Raises
        ------
        StopIteration
            When there are no more batches to return.
        """
        next_index = self._subset_iterator.next()
        # If the dataset is incompatible with the new interface, fall back to
        # the old one
        assert hasattr(self._dataset, 'get')
        rval = self._next(next_index)

        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

    def _next(self, next_index):
        return tuple(
            fn(batch) if fn else batch for batch, fn in
            safe_izip(self._dataset.get(self._source, next_index),
                      self._convert)
        )

    def __next__(self):
        return self.next()

    @property
    @wraps(SubsetIterator.batch_size, assigned=(), updated=())
    def batch_size(self):
        return self._subset_iterator.batch_size

    @property
    @wraps(SubsetIterator.num_batches, assigned=(), updated=())
    def num_batches(self):
        return float('inf')

    @property
    @wraps(SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        return float('inf')

    @property
    @wraps(SubsetIterator.uneven, assigned=(), updated=())
    def uneven(self):
        return False

    @property
    @wraps(SubsetIterator.stochastic, assigned=(), updated=())
    def stochastic(self):
        return self._subset_iterator.stochastic


class CosDataset(Dataset):

    """
    A dataset that streams randomly generated 2D examples.

    The first coordinate is sampled from a uniform distribution.
    The second coordinate is the cosine of the first coordinate,
    plus some gaussian noise.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, min_x=-6.28, max_x=6.28, std=.05, rng=_default_seed):
        """
        Constructor.
        """
        super(CosDataset, self).__init__()
        self.min_x = min_x
        self.max_x = max_x
        self.std = std

        # argument to resolve_iterator_class() can be either
        # a string from [sequential, shuffled_sequential, random_slice,
        # random_uniform, batchwise_shuffled_sequential, even_sequential,
        # even_shuffled_sequential, even_batchwise_shuffled_sequential,
        # even_sequences] or a SubsetIterator sublass.
        self._iter_subset_class = resolve_iterator_class('sequential')
        self._iter_data_specs = (VectorSpace(2), 'features')
        self._iter_batch_size = 100
        self._iter_num_batches = float('inf')
        self.rng = make_np_rng(rng, which_method=['uniform', 'randn'])


    def get (self, source, next_index):
        """
        next_index is a slice that tells how many examples are requested.
        """
        out_size = len(source)
        x = np.cast[config.floatX](self.rng.uniform(self.min_x, self.max_x,
                                                    (out_size, 1)))
        y = np.cos(x) + (np.cast[config.floatX](self.rng.randn(*x.shape)) *
                         self.std)
        result = np.reshape(y, (out_size, 1))
        return result

    @wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        # we can't use modes that require the size of the dataset
        orig_mode = mode
        if mode in _banned_mods:
            raise ValueError('Mode %s is not supported in CosDataset' % mode)
        if mode == 'random':
            mode = RandomSubsetIterator
        [mode, batch_size, num_batches, rng, data_specs] = self._init_iterator(
            mode, batch_size, num_batches, rng, data_specs)
        # sequential mode does not allow a random number generator
        if orig_mode == 'sequential':
            rng = None
        return InfiniteDatasetIterator(self,
                                       mode(float('inf'),
                                            batch_size,
                                            num_batches,
                                            rng),
                                       data_specs=data_specs,
                                       return_tuple=return_tuple)

    @wraps(Dataset.adjust_for_viewer)
    def adjust_for_viewer(self, X):
        return X

    @wraps(Dataset.has_targets)
    def has_targets(self):
        return True

    @wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return float('inf')
