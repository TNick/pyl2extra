"""
Classes that provide files and classes to the ImgDataset.
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
import logging
import numpy
import os
from pylearn2.utils import as_floatX
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng

class Provider(object):
    """
    Abstract class providing the interface for data providers.

    If implementations below are not suited for your needs implement a
    subclass of this class. The methods to be implemented are ``next()``
    and ``category()``.
    """
    def __init__(self):
        super(Provider, self).__init__()

    def next(self):
        """
        Either gets the name of next file or throws StopIteration.

        Returns
        -------
        fpath : str
            Path for next file to be considered.
        """
        raise NotImplementedError()

    def __next__(self):
        """
        Get next file.

        Returns
        -------
        fpath : str
            Path for next file to be considered.
        """
        return self.next()

    def cnext(self):
        """
        Gets next file; if the end is reached first file is returned
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Number of files stored inside.

        Returns
        -------
        len : int
            A non-negative integer.
        """
        return len(self.everything())

    def __iter__(self):
        """
        Standard interface for iterators.
        """
        return self

    def setup(self, dataset):
        """
        Called by the dataset once it initialized itself.
        """
        pass
        #assert isinstance(dataset, ImgDataset)

    def tear_down(self):
        """
        Called by the dataset fromits tear_down() method.
        """
        pass

    @staticmethod
    def normalize_image(img):
        """
        Converts an image into the format used by ``read()``.
        """
        if img.mode == 'LAB' or img.mode == 'HSV':
            raise ValueError('%s image mode is not supported' % img.mode)
        img = img.convert('RGBA')
        imarray = as_floatX(numpy.array(img)) / 255.0
        assert numpy.all(imarray >= 0.0) and numpy.all(imarray <= 1.0)
        assert len(imarray.shape) == 3
        assert imarray.shape[2] == 4
        return imarray

    def read(self, f_path):
        """
        Read a file and return an image and its category as a tuple.

        This method only reads internal images. For a more general method
        see `read_image()`.

        The returned array has (0, 1, 'c') shape, with c always 4
        (r, g, b, a) and datatype being floatX.

        Parameters
        ----------
        f_path : str
            The path as returned by a previous ``next()`` call.

        Returns
        -------
        result : tuple
            First member is the image as an numpy array,
            second is the category.
        """
        categ = self.category(f_path)
        logging.debug('generator reading [%s] from %s', categ, f_path)
        img = Image.open(f_path)
        return Provider.normalize_image(img), categ

    def read_image(self, image, internal=True):
        """
        Read an image and convert it in the format used by Provider.

        Parameters
        ----------
        image : str or Image.Image or numpy.ndarray
            The path for the image or the image itself.
        internal : bool
            Interpretation of the ``image`` parameter if it is a string:
            - If ``internal`` is True the image is read
              using `Provider.read()` (it is  assumed it is
              one of dataset's own image).
            - If False it is interpreted as being a
              path and is read using Image.

        The returned array is normalized to (0, 1, 'c') shape, with c always 4
        (r, g, b, a) and datatype being floatX.

        Returns
        -------
        result : numpy.ndarray
            The resulted image.
        """
        if isinstance(image, basestring):
            if internal:
                image, categ = self.read(image)
            else:
                image = Image.open(image)
                image = Provider.normalize_image(image)
        elif isinstance(image, numpy.ndarray):
            assert len(image.shape) == 3
            image = as_floatX(image)
        else:
            assert isinstance(image, Image.Image)
            image = Provider.normalize_image(image)
        # image is an array
        if image.shape[2] == 3:
            app = numpy.ones(shape=(image.shape[0], image.shape[1], 1),
                             dtype=image.stype) * 255
            numpy.append(image, app, axis=2)
        assert numpy.all(image >= 0.0) and numpy.all(image <= 1.0)
        return image

    def category(self, f_path):
        """
        Given a file name returned by next finds the category for that file.

        Parameters
        ----------
        iterator_cls : class
            An iterator class that inherits from SubsetIterator

        Returns
        -------
        categ : str
            A string indicating the category.
        """
        raise NotImplementedError()

    def categ_len(self):
        """
        Number of categories for this Provider.
        """
        return len(self.categories())

    def categories(self):
        """
        Get the list of categories.

        The list should always be the same for the lifetime of the instance
        as the indices are used to identify the class.

        Returns
        -------
        categs : list of str
            A string indicating the category.
        """
        if hasattr(self, 'categs'):
            return self.categs
        else:
            return self.get_categs()

    def get_categs(self):
        """
        Get a list of all categories in this provider.

        The implementation in the base class caches the result so, in you
        later want to insert images make sure to update the category
        yourself.
        """
        self.categs = []
        for key in self:
            ctg = self.category(key)
            if not ctg in self.categs:
                self.categs.append(ctg)
        return self.categs

    def categ2int(self, categ):
        """
        Converts a string representing a category name into an integer.

        Parameters
        ----------
        categ : str
            A category name.

        Returns
        -------
        idx : int
            The index of this category in the list of categories.
        """
        idx = self.categories().index(categ)
        if idx == -1:
            raise ValueError('%s is not among the '
                             'categories of the Provider' % categ)
        return idx

    def int2categ(self, idx):
        """
        Converts a string representing a category name into an integer.

        Parameters
        ----------
        idx : int
            The numeric id of the catgory.

        Returns
        -------
        categ : str
            The category coresponding to provided integer..
        """

        ctgs = self.categories()
        if idx < 0 or idx >= len(ctgs):
            raise ValueError('%d is outside the valid range '
                             'for categories of the Provider' % idx)
        return ctgs[idx]

    def everything(self):
        """
        The whole set of files and categories as a dictionary.

        Returns
        -------
        retdict : dict
            A dictionary where keys are files and values are categories.
        """
        retdict = {}
        for key in self:
            retdict[key] = self.category(key)
        return retdict

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        hdict = hash(self.__class__.__name__)
        evr = self.everything()
        for k in evr:
            hdict = hdict ^ (hash(k) << 1) ^ (hash(evr[k]) << 2)
        return  hdict

    def get(self, offset, count):
        """
        Get the files in a given range.

        The method does not throw errors if the range is outside allowed
        range. instead, the list of files is treated as a ring.

        Parameters
        ----------
        offset : int
            Zero-based indes of the first file to retreive.
        count : int
            Number of files to retreive.
        """
        assert offset >= 0
        assert count > 0
        evrt = self.everything()
        evrt_len = len(evrt)
        keys = evrt.keys()

        result = []
        while count > 0:
            # bring offset in valid range
            offset = offset % evrt_len
            # max number of items
            valid_range = evrt_len - offset
            # how many are we going to get on this round
            this_round = min(valid_range, count)
            result.extend(keys[offset:offset+this_round])

            offset = offset + this_round
            count = count - this_round

        return result


class DictProvider(Provider):
    """
    A provider based on a dictionary.

    Parameters
    ----------
    data : dict of strings
        A dictionary with the keys being file paths and values being
        the category.
    """
    def __init__(self, data):
        super(DictProvider, self).__init__()

        #: The dictionary that maps paths to classes.
        self.data = data

        #: The list of paths used for iteration purposes.
        self.keys_iter = data.keys().__iter__()

    @functools.wraps(Provider.__iter__)
    def __iter__(self):
        return self.data.keys().__iter__()

    @functools.wraps(Provider.__iter__)
    def __len__(self):
        return len(self.data)

    @functools.wraps(Provider.next)
    def next(self):
        return self.keys_iter.next()

    @functools.wraps(Provider.category)
    def category(self, f_path):
        return self.data[f_path]

    @functools.wraps(Provider.everything)
    def everything(self):
        return self.data

    @functools.wraps(Provider.cnext)
    def cnext(self):
        try:
            return self.next()
        except StopIteration:
            self.keys_iter = self.data.keys().__iter__()
            return self.next()

    def __getstate__(self):
        """
        Help pickle this instance.
        """
        if not hasattr(self, 'categs'):
            self.get_categs()
            assert hasattr(self, 'categs')

        return {'data': self.data, 'categs': self.categs}

    def __setstate__(self, state):
        """
        Help un-pickle this instance.
        """
        self.data = state['data']
        self.categs = state['categs']

        #: The list of paths used for iteration purposes.
        self.keys_iter = self.data.keys().__iter__()

class CsvProvider(DictProvider):
    """
    A provider based on a comma separated values file.

    The constructor will read the file and construct a dictionary that is
    later used by the DictProvider superclass.

    Parameters
    ----------
    csv_path : str
        The path where the csv file can be found.
     : bool
        Skip first row if true. ``has_header`` must be True if ``col_path``
        or ``col_class`` are strings.
    col_path : str or int, optional
        If this is an integer, it represents the zero based index of the column
        that contains image paths.
        If this is a string, it represents the name of the column; the name
        will be searched in the header (so it is asserted the ``has_header``
        is true).
    col_class : str or int, optional
        If this is an integer, it represents the zero based index of the column
        that contains image paths.
        If this is a string, it represents the name of the column; the name
        will be searched in the header (so it is asserted the ``has_header``
        is true).
    delimiter : str, optional
        A one-character string used to separate fields. It defaults to ``,``.
    quotechar : str, optional
        A one-character string used to quote fields containing special
        characters, such as the delimiter or quotechar,
        or which contain new-line characters. It defaults to ``"``.
    skip_first : int, optional
        Skip this many rows at the beginning of the file; this does
        not include the header row that is managed separatelly.
    skip_last : int, optional
        Skip this many rows at the end of the file.

    Notes
    -----
    The class is only interested in two columns: the class and the path.
    Both results are passed through pylearn2.utils.string_utils.preprocess
    so you can use environment variables to customize their final value.

    The paths inside csv file, if relative, are considered relative to
    the path of the .csv file.

    For the last entries to be removed the entries are parsed
    and added to a list. They are later removed from the result.
    This is in order to avoid reading the csv file twice but
    it has the downside that the entries to be skipped stil
    have to be valid (enough columns).
    """
    def __init__(self, csv_path, col_path=1, col_class=0, has_header=False,
                 delimiter=',', quotechar='"',
                 skip_first=0, skip_last=0):

        assert skip_first >= 0
        assert skip_last >= 0

        if (isinstance(col_class, basestring) or
                isinstance(col_path, basestring)):
            # we will need to locate the indices so we need a header
            assert has_header
        else:
            col_min = max(col_path, col_class)

        # make the path absolute and extract base directory
        csv_path = os.path.abspath(csv_path)
        csv_base = os.path.split(csv_path)[0]

        # collect data in a dictionary
        data = {}
        file_list = []
        with open(csv_path, 'rt') as fhand:
            csvr = csv.reader(fhand,
                              delimiter=delimiter,
                              quotechar=quotechar)

            # collect all rows
            for i, row in enumerate(csvr):
                if len(row) == 0:
                    continue
                if has_header:
                    # find the actual indices for columns of interest
                    if isinstance(col_class, basestring):
                        col_class = row.index(col_class)
                    if isinstance(col_path, basestring):
                        col_path = row.index(col_path)
                    has_header = False
                    col_min = max(col_path, col_class)
                    skip_first = skip_first + 1
                    continue

                if i < skip_first:
                    continue

                if len(row) <= col_min:
                    err = '%s[%d] should have at least %d ' + \
                          'columns but it only has %d'
                    raise ValueError(err % (csv_path, i, col_min, len(row)))
                else:
                    class_name = preprocess(row[col_class]).strip().lower()
                    fpath = preprocess(row[col_path])
                    if len(fpath) > 0:
                        if not os.path.isabs(fpath):
                            fpath = os.path.join(csv_base, fpath)
                    data[fpath] = class_name
                    if skip_last > 0:
                        file_list.append(fpath)

        # remove last entries
        if skip_last > 0:
            if len(file_list) <= skip_last:
                data = {}
            else:
                for f2rem in file_list[-skip_last:]:
                    data.pop(f2rem, None)

        # everything else is provided by DictProvider
        super(CsvProvider, self).__init__(data)


class RandomProvider(DictProvider):
    """
    A provider that has 100 random images labelled 1 to 100

    The constructor will create the random images and will construct
    a dictionary that is used by the DictProvider superclass.

    Parameters
    ----------
    rng : `numpy.random.RandomState` or seed, optional
        Seed for random number generator or an actual RNG.
    """
    def __init__(self, rng=None, count=100, alpha=False, size=(128, 128)):
        self.rng = make_np_rng(rng)
        self.content = {}
        self.rng = rng
        self.count = count
        self.alpha = alpha
        self.size = size
        data = self.prepare()

        # everything else is provided by DictProvider
        super(RandomProvider, self).__init__(data)

    @functools.wraps(Provider.__iter__)
    def __len__(self):
        return self.count

    @functools.wraps(Provider.read)
    def read(self, f_path):
        logging.debug('generator reading file %s', f_path)
        categ = f_path
        img = self.content[f_path]
        return Provider.normalize_image(img), categ

    def __getstate__(self):
        """
        Help pickle this instance.
        """
        state = {}
        state['count'] = self.count
        state['alpha'] = self.alpha
        state['size'] = self.size
        return state

    def __setstate__(self, state):
        """
        Help un-pickle this instance.
        """
        self.count = state['count']
        self.alpha = state['alpha']
        self.size = state['size']
        self.data = self.prepare()
        self.keys_iter = self.data.keys().__iter__()

    def prepare(self):
        """
        Initializes the instance according to internal attributes.
        """
        self.content = {}
        data = {}
        modes = ['RGBA', 'RGB', '1', 'L', 'P', 'CMYK', 'I', 'F']
        channels = 4 if self.alpha else 3
        for i in range(self.count):
            file_name = str(i+1)
            imarray = numpy.random.rand(self.size[0],
                                        self.size[1],
                                        channels) * 255
            img = Image.fromarray(imarray.astype('uint8'))
            self.content[file_name] = img.convert(modes[i % len(modes)])
            data[file_name] = file_name
        return data


class DeDeMaProvider(Provider):
    """
    A provider based on a DenseDesignMatrix.

    Parameters
    ----------
    data : dict of strings
        A dictionary with the keys being file paths and values being
        the category.
    """
    def __init__(self, dedema):
        self.dedema = dedema
        self.dataset_iter = range(dedema.get_num_examples()).__iter__()
        super(DeDeMaProvider, self).__init__()

    @functools.wraps(Provider.__iter__)
    def __iter__(self):
        return self.dataset_iter

    @functools.wraps(Provider.__iter__)
    def __len__(self):
        return self.dedema.get_num_examples()

    @functools.wraps(Provider.next)
    def next(self):
        return self.dataset_iter.next()

    @functools.wraps(Provider.category)
    def category(self, f_path):
        f_path = int(f_path)
        assert not self.dedema.y is None
        return self.dedema.y[f_path]

    @functools.wraps(Provider.everything)
    def everything(self):
        rang = range(self.dedema.get_num_examples())
        data = {}
        for i in rang:
            data[i] = self.dedema.y[i]
        return data

    @functools.wraps(Provider.cnext)
    def cnext(self):
        try:
            return self.next()
        except StopIteration:
            rang = range(self.dedema.get_num_examples())
            self.dataset_iter = rang.__iter__()
            return self.next()

    def __getstate__(self):
        """
        Help pickle this instance.
        """
        return {'data': self.dedema}

    def __setstate__(self, state):
        """
        Help un-pickle this instance.
        """
        self.dedema = state['data']
        self.dataset_iter = self.dedema.iterator()

    @functools.wraps(Provider.read)
    def read(self, f_path):
        f_path = int(f_path)
        categ = self.category(f_path)
        ddm = self.dedema.get_design_matrix()
        view_conv = self.dedema.view_converter.design_mat_to_topo_view
        exm = view_conv(ddm[f_path].reshape(1, ddm.shape[1]))
        return exm[0], categ
