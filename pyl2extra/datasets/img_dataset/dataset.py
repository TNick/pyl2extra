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
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import FiniteDatasetIterator
from pylearn2.utils import as_floatX
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
                                 "cache_loc a path towards an existing, "
                                 "writable directory that stores the cache; "
                                 "the user provided a %s" %
                                 str(cache_loc.__class__))
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

        #: the number of images that results from a single input image
        self.transf_count = 1
        for adj in self.adjusters:
            self.transf_count = self.transf_count * adj.transf_count()
        self.transf_count = int(self.transf_count)

        #: total number of unique examples
        self.totlen = len(self.data_provider) * self.transf_count
        empty = self.totlen == 0
        assert empty or (self.totlen > 0)

        # Dataset._init_iterator offers a dataset the oportunity to
        # provide defaults for iterator-related values:

        #: default class (NOT instance) for iterating the dataset; this
        # can NOT be a string from pylearn2.utils.iteration._iteration_schemes
        self._iter_subset_class = SequentialSubsetIterator

        #: default number of examples in a batch
        self._iter_batch_size = 128

        #: default number of batches
        totrawex = len(self.data_provider)
        self._iter_num_batches = totrawex / self._iter_batch_size
        if totrawex > self._iter_num_batches * self._iter_batch_size:
            self._iter_num_batches = self._iter_num_batches  + 1

        #: default random number generator
        self.rng = make_np_rng(rng_or_seed=rng,
                               default_seed=[2017, 5, 17],
                               which_method=["random_integers"])

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

        # self._init_iterator will return the default number of batches that
        # matches the default batch size; we need a number that reflects batch_size
        # argument.
        totrawex = len(self.data_provider)
        num_batches = totrawex / batch_size
        #if totrawex > num_batches * batch_size:
        #    num_batches = num_batches + 1

        return FiniteDatasetIterator(self,
                                     mode(num_batches * batch_size,
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
        # we can't return len(self) because the number is used by the iterator
        # (and, as such, the monitor) which is going to run humongous epocs
        return len(self.data_provider)

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

    def process(self, batch, accumulate=False):
        """
        Runs the batch through all adjusters and returns the result.

        Parameters
        ----------
        batch : numpy.array
            An array to process. The expected shape is ('b', W, H, 'c'),
            with c being 4: red, green, blue and alpha
        accumulate : bool, optional
            Wheter to gnerate all possible combinations (True) or to
            generate a single image.

        Returns
        -------
        batch : numpy.array
            The resulted batch, processed. The result's shape is
            ('b', W, H, 'c'), with c being 3: red, green, blue (no alpha).
        """
        assert len(batch.shape) == 4
        assert batch.shape[3] == 4
        result = as_floatX(batch)
        if len(self.adjusters) > 0:
            if accumulate:
                for adj in self.adjusters:
                    result = adj.accumulate(result)
            else:
                for adj in self.adjusters:
                    result = adj.process(result)
        else:
            result = result[:, :, :, :3]
        #if result.shape[3] != self.channels_len():
        #    raise AssertionError("Adjusters must be organized in such a way "
        #                         "that at the end of the processing cycle "
        #                         "the result is in RGB form (as opposed to "
        #                         "original RGBA form). One common way for "
        #                         "doing that is to use BackgroundAdj.")
        return result

    def process_labels(self, size):
        """
        Gets the textual description for the transformations.

        This method can be used to describe what transformations a
        particular image suffered when subjected to Adjuster.accumulate()
        chain (calling `process(batch, accumulate=True)`).

        Parameters
        ----------
        size : int
            The length of the batch that was passed to `process()`.

        Returns
        -------
        result : list
            A list of items, each one representing an image. An imtem is
            represented as a list of dictionaries, one for easch adjuster.
            Inside each dictionary a key-value pair is present for each
            parameter.
        """
        result = [[] * size]
        for adj in self.adjusters:
            result = adj.accum_text(result)
        return result

    def get_cache_loc(self):
        """
        Get th location of the cache for this dataset.
        Returns
        -------
        path : str
            None if caching is disabled or the path towards the cache directory.
        """
        if self.cache_loc is None:
            return None
        path = os.path.join(self.cache_loc, 'h' + str(hash(self)))
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    def get_topo_batch_axis(self):
        """
        The index of the axis of the batches.

        This is the same method as ``DenseDesignMatrix.get_topo_batch_axis()``
        in ``pylearn2.datasets.dense_design_matrix``.

        Returns
        -------
        axis : int
            The axis of a topological view of this dataset that corresponds
            to indexing over different examples.
        """
        axis = self.axes.index('b')
        return axis

    def get_topological_view(self, mat=None):
        """
        Convert an array (or the entire dataset) to a topological view.

        This is the same method as ``DenseDesignMatrix.get_topological_view()``
        in ``pylearn2.datasets.dense_design_matrix``.

        Parameters
        ----------
        mat : ndarray, 2-dimensional, optional
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.
            This parameter is not named X because X is generally used to
            refer to the design matrix for the current problem. In this
            case we want to make it clear that `mat` need not be the design
            matrix defining the dataset.
        """
        if mat is None:
            mat = self.to_dense_design_matrix().X
        assert mat.shape[1] == self.shape[0] * self.shape[1] * 3
        return numpy.reshape(mat, mat.shape[0],
                             self.shape[0], self.shape[1], 3)

    def get_design_matrix(self, topo=None):
        """
        Return topo (a batch of examples in topology preserving format),
        in design matrix format.

        This is the same method as ``DenseDesignMatrix.get_design_matrix()``
        in ``pylearn2.datasets.dense_design_matrix``.

        Parameters
        ----------
        topo : ndarray, optional
            An array containing a topological representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.
        Returns
        -------
        WRITEME
        """
        if topo is None:
            topo = self.to_dense_design_matrix().X
        return numpy.reshape(topo, topo.shape[0],
                             topo.shape[1] * topo.shape[2] * topo.shape[3])

    def get_targets(self):
        """
        Get the classes for all examples.
        """
        return self._to_dense_design_matrix()[1]

    def _to_dense_design_matrix(self, num_examples=None, ne_raw=True):
        """
        Generates a DenseDesignMatrix from examples in this dataset.
        """
        if num_examples is None:
            num_examples = self.totlen
            ne_raw = False
        elif ne_raw == True:
            num_examples = num_examples * self.transf_count
        # num_examples now represents the number f examples in final dataset

        result_x = None
        result_y = None
        ofs = 0
        for fpath in self.data_provider:
            # generate all combinations from this image
            batch = self.data_provider.read_image(image=fpath,
                                                  internal=True)
            batch = batch.reshape(1, batch.shape[0],
                                  batch.shape[1], batch.shape[2])
            batch = self.process(batch, accumulate=True)
            clss = self.data_provider.category(fpath)
            classes = [clss for i in range(batch.shape[0])]

            # save it into our array
            if result_x is None:
                shape = (num_examples, self.shape[1], self.shape[0], 3)
                result_x = numpy.empty(shape=shape, dtype=batch.dtype)
                result_y = []
            to_copy = min(num_examples, batch.shape[0])
            result_x[ofs:ofs+to_copy, :, :, :] = batch[0:to_copy, :, :, :]
            result_y += classes[0:to_copy]

            # update counters
            ofs = ofs + to_copy
            num_examples = num_examples - to_copy
            if num_examples <= 0:
                break

        result_y = numpy.array(result_y).reshape(len(result_y), 1)
        result = DenseDesignMatrix(topo_view=result_x,
                                   y=result_y,
                                   axes=('b', 0, 1, 'c'),
                                   preprocessor=None,
                                   fit_preprocessor=False,
                                   X_labels=None, y_labels=None)
        return result, result_y

    def to_dense_design_matrix(self, num_examples=None, ne_raw=True):
        """
        Generates a DenseDesignMatrix from examples in this dataset.

        Parameters
        ----------
        num_examples : int, optional
            Number of examples to place in the new dataset. The meaning of
            this parameter depends on ``ne_raw``. If None all examples are
            going to be processed and saved. Note that this may be a
            large number and that the whole resulted dataset needs
            to fit in memory.
        ne_raw : bool, optional
            How to interpret ``num_examples``. if True, ``num_examples``
            represents the number of raw examples to process; final number of
            examples in the DenseDesignMatrix instance will be
            ``num_examples`` times number of processing steps. If False,
            ``num_examples`` represents the desired number of examples
            in output dataset.

        Returns
        -------
        ddm : DenseDesignMatrix
            A topology-preserving dense dataset. The number of channels is
            always 3.
        """
        return self.to_dense_design_matrix(num_examples, ne_raw)[0]

