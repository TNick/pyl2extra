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
import numpy
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

    def normalize_image(self, im):
        """
        Converts an image into the format used by ``read()``.
        """
        if im.mode == 'LAB' or im.mode == 'HSV':
            raise ValueError('%s image mode is not supported' % im.mode)
        im = im.convert('RGBA')
        imarray = as_floatX(numpy.array(im))
        assert len(imarray.shape) == 3
        assert imarray.shape[2] == 4
        return imarray

    def read(self, f_path):
        """
        Read a file and return an image and its category as a tuple.

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
        im = Image.open(f_path)
        return self.normalize_image(im), categ

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

    @functools.wraps(Provider.category)
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
    """
    def __init__(self, csv_path, col_path=1, col_class=0, has_header=False,
                 delimiter=',', quotechar='"'):

        if (isinstance(col_class, basestring) or
                isinstance(col_path, basestring)):
            # we will need to locate the indices so we need a header
            assert has_header
        else:
            col_min = max(col_path, col_class)

        # collect data in a dictionary
        data = {}
        with open(csv_path, 'rt') as fhand:
            csvr = csv.reader(fhand,
                              delimiter=delimiter,
                              quotechar=quotechar)
            # collect all rows
            for i, row in enumerate(csvr):
                if has_header:
                    # find the actual indices for columns of interest
                    if isinstance(col_class, basestring):
                        col_class = row.index(col_class)
                    if isinstance(col_path, basestring):
                        col_path = row.index(col_path)
                    has_header = False
                elif len(row) < col_min:
                    err = '%s[%d] should have at least %d ' + \
                          'columns but it only has %d'
                    raise ValueError(err % (csv_path, i, col_min, len(row)))
                else:
                    ppd = preprocess(row[col_class])
                    data[preprocess(row[col_path])] = ppd

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

    @functools.wraps(Provider.read)
    def read(self, f_path):
        categ = f_path
        im = self.content[f_path]
        return self.normalize_image(im), categ

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
        self.content = {}
        data = {}
        modes = ['RGBA', 'RGB', '1', 'L', 'P', 'CMYK', 'I', 'F']
        channels = 4 if self.alpha else 3
        for i in range(self.count):
            file_name = str(i+1)
            imarray = numpy.random.rand(self.size[0],
                                        self.size[1], 
                                        channels) * 255
            im = Image.fromarray(imarray.astype('uint8'))
            self.content[file_name] = im.convert(modes[i % len(modes)])
            data[file_name] = file_name
        return data
