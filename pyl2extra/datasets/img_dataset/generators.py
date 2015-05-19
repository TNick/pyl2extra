"""
Classes that generate content for ImgDataset.

These classes deal with the distributin of labour. The work may be done
online, in worker threads or in owrker processes.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import multiprocessing
import numpy
#from pyl2extra.datasets.img_dataset.dataset import ImgDataset

from pyl2extra.utils import slice_count

class Generator(object):
    """
    The class is used to generate content.
    """
    def __init__(self):

        #: associated dataset - the bound is created in setup() method
        self.dataset = None

        super(Generator, self).__init__()

    def is_inline(self):
        """
        Tell if this generator works on the same thread as the requester.

        Returns
        -------
        inline : bool
            True if the thread will block waiting for the result, False if
            the result is generated in paralel.
        """
        raise NotImplementedError()

    def setup(self, dataset):
        """
        Called by the dataset once it initialized itself.
        """
        self.dataset = dataset
        #assert isinstance(dataset, ImgDataset)

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        return hash(self.__class__.__name__)

    def get(self, source, next_index):
        """
        The get method used by the dataset to retreive batches of data.

        Parameters
        ----------
        source : touple of str
            A tuple of source identifiers (strings) to indicate the
            source for the data to retreive. The iterator will receive
            a ``data_specs`` argument consisting of ``(space, source)``.
        next_index : list or slice object
            The indexes of the examples to retreive specified either as a
            list or as a slice.

        Returns
        -------
        next_batch : tuple
            The result is a tuple of batches, one for each ``source`` that
            was requested. Each batch in the tuple should follow the
            dataspecs for the dataset.
        """
        raise NotImplementedError()

    def _prep_get(self, source, next_index):
        """
        Common opperations for a get() call.
        """
        count = slice_count(next_index)

        # prepare for iteration
        idx_features = -1
        idx_targets = -1
        result = []
        for i, src in enumerate(source):
            if src == 'features':
                idx_features = i
                result.append(numpy.zeros(shape=(count,
                                                 self.dataset.shape[0],
                                                 self.dataset.shape[1],
                                                 3)))
            elif src == 'targets':
                idx_targets = i
                result.append(numpy.zeros(shape=(count, 1), dtype='int32'))
            else:
                raise ValueError('%s implements <features> and <targets>; '
                                 '<%s> is not among these.' %
                                 (str(self.__class__.__name__), src))
        return count, result, idx_features, idx_targets


class InlineGen(Generator):
    """
    Generates the content while the other parties wait on the same thread.
    """
    def __init__(self):
        super(InlineGen, self).__init__()

    @functools.wraps(Generator.is_inline)
    def is_inline(self):
        return True

    @functools.wraps(Generator.is_inline)
    def get(self, source, next_index):
        count, result, idx_features, idx_targets = self._prep_get(source,
                                                                  next_index)
        # iterate to collect data
        for i in range(count):
            fpath = self.dataset.data_provider.cnext()
            trg, categ = self.dataset.data_provider.read(fpath)
            categ = self.dataset.data_provider.categ2int(categ)
            trg = numpy.reshape(trg,
                                (1, trg.shape[0], trg.shape[1], trg.shape[2]))
            if idx_features > -1:
                trg = self.dataset.process(trg)
                result[idx_features][i, :, :, :] = trg
            if idx_targets > -1:
                result[idx_targets][i][0] = categ

        return tuple(result)


class ThreadedGen(Generator):
    """
    Generates the content using separate threads in same process.

    Parameters
    ----------
    count : int, optional
        The number of worker threads to use. If None, same number of threads
        as the number of cores minus one are used.
    """
    def __init__(self, count=None):

        if count is None:
            count = multiprocessing.cpu_count()
            count = count - 1 if count > 1 else 1
        elif count < 0:
            raise ValueError("Number of processes must be a positive integer")
        #: number of workers to use
        self.count = count

        super(ThreadedGen, self).__init__()

    @functools.wraps(Generator.is_inline)
    def is_inline(self):
        return False

    @functools.wraps(Generator.is_inline)
    def get(self, source, next_index):
        count, result, idx_features, idx_targets = self._prep_get(source,
                                                                  next_index)

        # look into the cache

        # not found, so wait for it

        return tuple(result)


class ProcessGen(Generator):
    """
    Generates the content using separate processes.

    Parameters
    ----------
    count : int, optional
        The number of worker processes to use. If None, same number of
        processes as the number of cores minus one are used.
    """
    def __init__(self, count=None):

        if count is None:
            count = multiprocessing.cpu_count()
            count = count - 1 if count > 1 else 1
        elif count < 0:
            raise ValueError("Number of processes must be a positive integer")
        #: number of workers to use
        self.count = count



        super(ProcessGen, self).__init__()

    @functools.wraps(Generator.is_inline)
    def is_inline(self):
        return False


def genFromString(gen_name):
    """
    Creates a generator based on a string key.

    Parameters
    ----------
    gen_name : str
        A string identifying the type of Generator to use.

    Returns
    -------
    adj : Generator
        The instance that was constructed.
    """

    if gen_name == 'inline':
        return InlineGen()
    elif gen_name == 'threads':
        return ThreadedGen()
    elif gen_name == 'process':
        return ProcessGen()
    else:
        raise ValueError('%s is not a known Generator name' % gen_name)
