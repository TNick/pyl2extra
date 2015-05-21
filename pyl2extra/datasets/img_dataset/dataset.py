"""
Dataset for img_dataset momdule.

In these modules a preprocessor is names an adjuster to avoid the confusion
between these and pylearn2 preprocessors.
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
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import FiniteDatasetIterator
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils.iteration import SequentialSubsetIterator

from pylearn2.space import Conv2DSpace, CompositeSpace, IndexSpace

from pyl2extra.datasets.img_dataset.data_providers import (Provider,
                                                           DictProvider,
                                                           CsvProvider)
from pyl2extra.datasets.img_dataset.adjusters import (Adjuster,
                                                      adj_from_string)
from pyl2extra.datasets.img_dataset.generators import (Generator,
                                                       InlineGen,
                                                       gen_from_string)


class ImgDataset(Dataset):
    """
    A dataset that can be preprocessed as it is being trained.

    Parameters
    ----------
    data_provider : str, dictionary or Provider
        If it is a string it is interpreted as being a csv file with
        first column having the class and second column the path towards
        the image.
        If it is a dictionary the keys are file paths and the values
        are the classes.
        Both string and dictionary variant are internally converted to
        coresponding Provider class.
    adjusters : list of Adjuster or strings, optional
        The list can consist of either actual adjusters or names for them.
        The constructor will initialize the entries consisting of strings with
        appropriate adjuster using default arguments.
    generator : str or Generator, optional
        The option can be a Generator subclass or a name for one.
        If later default values will be used with the constructor.
        By default an inline generator is used.
    cache_loc : str, optional
        The path towards the cache location. If None caching is disabled.
        Caching can be very useful when a dataset is used repeatedly. Most
        of the time the dataset has plenty of time to generate content
        while previous batch is used for training. However, for the first
        batch, the network will have to wait as there is no previous batch.
        If a cache_location is present the network will look for first batch
        from a previous run and will use it as the first batch if found.
    rng : `numpy.random.RandomState` or seed, optional
        The random number generator associated with this dataset
    """
    def __init__(self, data_provider, adjusters=None, generator=None,
                 shape=None, axes=None, cache_loc=None, rng=None):

        #: The shape of the image (may be overriden by Providers)
        self.shape = shape if shape else (128, 128)

        #: The format for batches (may be overriden by Providers)
        self.axes = self._check_axes(axes)

        #: data provider for files and categories
        self.data_provider = self._check_data_provider(data_provider)

        #: list of adjusters applied on the fly to the dataset
        self.adjusters = self._check_adjusters(adjusters)

        #: the generator that drives the process
        self.generator = self._check_generator(generator)

        if not cache_loc is None:
            if not os.path.isdir(cache_loc):
                raise ValueError("ImgDataset constructor accepts for its "
                                 "generator a path towards an existing, "
                                 "writable directory that stores the cache; "
                                 "the user provided a %s" %
                                 str(generator.__class__))
        #: location of the cache
        self.cache_loc = cache_loc

        #: the hash is used as a key for caching / storing things
        self.hash_value = hash(self.__class__.__name__) ^ \
            (hash(self.data_provider) << 2) ^ (hash(self.generator) << 1)
        for adj in self.adjusters:
            self.hash_value = self.hash_value ^ hash(adj)

        #: first batch that may have been precomputed by a previous run
        self.first_batch = None
        if self.cache_loc:
            cache_key = '%s/%x_first_batch.npy' % (self.cache_loc, hash(self))
            if os.path.isfile(cache_key):
                self.first_batch = numpy.load(cache_key)

        # intialize components
        self.data_provider.setup(self)
        for adj in self.adjusters:
            adj.setup(self, 'rand_one')
        self.generator.setup(self)

        #: total number of unique examples
        self.totlen = len(self.data_provider)
        empty = self.totlen == 0
        for adj in self.adjusters:
            self.totlen = self.totlen * adj.transf_count()
        assert empty or (self.totlen > 0)

        # Dataset._init_iterator offers a dataset the oportunity to
        # provide defaults for iterator-related values:

        #: default class (NOT instance) for iterating the dataset; this
        # can NOT be a string from pylearn2.utils.iteration._iteration_schemes
        self._iter_subset_class = SequentialSubsetIterator

        #: default number of examples in a batch
        self._iter_batch_size = 128

        #: default number of batches
        self._iter_num_batches = self.totlen / self._iter_batch_size
        if self.totlen > self._iter_num_batches * self._iter_batch_size:
            self._iter_num_batches = self._iter_num_batches  + 1

        #: default random number generator
        self.rng = make_np_rng(rng)

        #: default data specs
        self._iter_data_specs = self._data_specs()

        super(ImgDataset, self).__init__()

    def tear_down(self):
        """
        Done with the dataset. Calls `tear_down()` for all components.
        """
        # intialize components
        self.data_provider.tear_down()
        for adj in self.adjusters:
            adj.tear_down()
        self.generator.tear_down()

    def categ_len(self):
        """
        Number of categories.
        """
        return self.data_provider.categ_len()

    def categories(self):
        """
        List of categories.
        """
        return self.data_provider.categories()

    def channels_len(self):
        """
        Number of channels for images provided by this dataset.
        """
        return 3

    def _check_data_provider(self, data_provider):
        """
        Helps the constructor check data_provider
        """
        if isinstance(data_provider, dict):
            data_provider = DictProvider(data_provider)
        elif isinstance(data_provider, basestring):
            data_provider = CsvProvider(data_provider)
        elif not isinstance(data_provider, Provider):
            raise ValueError("ImgDataset constructor accepts for its "
                             "data_provider either a string (the path "
                             "to a csv file), a dictionary (keys are "
                             "image paths, values are classes) or a "
                             "Provider instance; the user provided a %s" %
                             str(data_provider.__class__))
        return data_provider

    def _check_generator(self, generator):
        """
        Helps the constructor check the generator
        """
        if generator is None:
            generator = InlineGen()
        elif isinstance(generator, basestring):
            generator = gen_from_string(generator)
        elif not isinstance(generator, Generator):
            raise ValueError("ImgDataset constructor accepts for its "
                             "generator a string (the name of a Generator) "
                             "or a Generator instance; "
                             "the user provided a %s" %
                             str(generator.__class__))
        return generator

    def _check_adjusters(self, adjusters):
        """
        Helps the constructor check the adjusters
        """
        if adjusters is None:
            adjusters = []
        for i, adj in enumerate(adjusters):
            if isinstance(adj, basestring):
                adjusters[i] = adj_from_string(adj)
            elif not isinstance(adj, Adjuster):
                raise ValueError("ImgDataset constructor accepts for its "
                                 "adjusters a list consisting of strings "
                                 "(adjuster names) or Adjuster instances; "
                                 "the user provided a %s" %
                                 str(adj.__class__))
        return adjusters

    def _check_axes(self, axes):
        """
        Helps the constructor check the axes.
        """
        def axtype(axes):
            if isinstance(axes, tuple):
                return True
            if isinstance(axes, list):
                return True
            return False
        if axes is None:
            axes = ('b', 0, 1, 'c')
        elif not axtype(axes) or len(axes) != 4:
            raise ValueError("ImgDataset constructor accepts for its "
                             "axes a tuple consisting of some "
                             "permutation of 'b', 0, 1, 'c'")
        return axes

    def _data_specs(self):
        """
        Helps the constructor generate data specs
        """
        ctg_cnt = self.data_provider.categ_len()
        x_space = Conv2DSpace(shape=self.shape,
                              num_channels=self.channels_len(),
                              axes=self.axes)
        x_source = 'features'
        if ctg_cnt == 0:
            space = x_space
            source = x_source
        else:
            y_space = IndexSpace(dim=1, max_labels=ctg_cnt)
            #y_space = VectorSpace(dim=ctg_cnt)
            y_source = 'targets'

            space = CompositeSpace((x_space, y_space))
            source = (x_source, y_source)

        return (space, source)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        [mode, batch_size, num_batches, rng, data_specs] = self._init_iterator(
            mode, batch_size, num_batches, rng, data_specs)
        return FiniteDatasetIterator(self,
                                     mode(len(self),
                                          batch_size,
                                          num_batches,
                                          rng),
                                     data_specs=data_specs)

    @functools.wraps(Dataset.adjust_for_viewer)
    def adjust_for_viewer(self, X):
        return X / numpy.abs(X).max()

    @functools.wraps(Dataset.has_targets)
    def has_targets(self):
        return self.data_provider.categ_len() > 0

    @functools.wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return len(self)

    def get(self, source, next_index):
        """
        The get method used by the iterators to retreive batches of data.

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
        return self.generator.get(source, next_index)

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        return self.hash_value

    def __len__(self):
        """
        Total number of examples.

        This is NOT the number of files that are used to generate content but
        total number of examples to be generated.

        The number of files used to generate content may be found by calling
        ``len(self.data_provider)``

        Returns
        -------
        totlen : int
            A non-negative integer.
        """
        return self.totlen

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.geta()` will be.

        The signature of this method is identical to that of
        DenseDesignMatrix.get_data_specs()
        """
        return self._iter_data_specs

    def process(self, batch):
        """
        Runs the batch through all adjusters and returns the result.

        Parameters
        ----------
        batch : numpy.array
            An array to process. The expected shape is ('b', W, H, 'c'),
            with c being 4: red, green, blue and alpha

        Returns
        -------
        batch : numpy.array
            The resulted batch, processed. The result's shape is
            ('b', W, H, 'c'), with c being 4: red, green, blue (no alpha).
        """
        assert len(batch.shape) == 4
        assert batch.shape[3] == 4
        result = batch
        if len(self.adjusters) > 0:
            for adj in self.adjusters:
                result = adj.process(result)
        else:
            result = result[:, :, :, :3]
        if result.shape[3] != self.channels_len():
            raise AssertionError("Adjusters must be organized in such a way "
                                 "that at the end of the processing cycle "
                                 "the result is in RGB form (as opposed to "
                                 "original RGBA form). One common way for "
                                 "doing that is to use BackgroundAdj.")
        return result

    def to_dense_design_matrix(self):
        """
        Generates all examples and returns them as a DenseDesignMatrix.


        Returns
        -------
        ddm : DenseDesignMatrix
            A topology-preserving dense dataset.
        """
        # TODO: implement
        raise NotImplementedError()
