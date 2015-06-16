"""
Dataset for images from a file system and related functionality.

"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from collections import OrderedDict
import numpy
import Image
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from theano.config import floatX


class Images(DenseDesignMatrix):
    """
    A pylearn2 dataset that loads the images from a list or csv file.

    Parameters
    ----------
    source : OrderedDict, dict, str, tuple, list
        This argument provides the input images and (optionally) 
        associated categories. The meaning of the argument depends
        on the data type:
        - if ``source`` is a string, it is interpreted to be the
          path towards a csv file; the file must NOT have a header,
          first column must contain the classes or values and
          second column must contain the paths for the image files;
        - if ``source`` is a dictionary, the keys must be the 
          paths for image files or ``Image`` instances and 
          the values must be the classes or values (None or empty 
          string if this instance does not provide the labels);
        - a tuple or list must have exactly one or two memobers:
          first one must be a list or tuple of image paths or
          Images, while second one (optional) has the classes
          or values.
    image_size : int, optional
        The size of the images in the final dataset. All images
        will be resized to be ``image_size`` x ``image_size``
        pixels.
    regression : bool, optional
        Tell if this is a classification problem (classes are
        expected to be integers) or a regression problem
        values should be convertible to real numbers.
    rng : object, optional
        A random number generator used for picking random \
        indices into the design matrix when choosing minibatches.
    preprocessor : Preprocessor, optional
        Preprocessor to apply to images.
    fit_preprocessor : bool, optional
        Whether preprocessor can fit parameters when applied to training
        data.    
    """
    def __init__(self, source, image_size=None, regression=False, 
                 rng=None,  preprocessor=None, fit_preprocessor=False):
        
        #: preserve original argument for future reference
        self.source = source

        #: regression or classification problem
        self.regression = regression

        if isinstance(source, basestring):
            # this is a csv file that we're going to read
            ind = self._load_dict(self._load_csv(source))
        elif isinstance(source, dict):
            # keys are file names, values are classes 
            ind = self._load_dict(source)
        elif isinstance(source, (list, tuple)):
            # one item lists the files, the other lists the classes
            if len(source) == 1:
                ind = self._load_dict(OrderedDict([(src, None) for src in source])
            elif len(source) == 2:
                if len(source[0]) == len(source[1]):
                    ind = self._load_dict(OrderedDict([pair 
                        for pair in zip(source[0], source[1])]))
                else:
                    raise ValueError("Lists/tuples provded to Images class constructor"
                                     "are expected to have same length (%d != %d)" %
                                     (len(source[0]), len(source[1])))

            else:
                raise ValueError("Lists/tuples provided to Images class constructor"
                                 "are expected to have one (images only) or two members "
                                 "(images and classes); the input has %d members." %
                                 len(source))
        else:
            raise ValueError("Images class expects for its `source` argument "
                             "a file path (string), a dictionary of file:class "
                             "pairs, or a pair of lists (tuples); %s is not "
                             "supported" % str(source.__class__))
        # all images are loaded as pil.Image in ``ind`` variable

        # DenseDesignMatrix expects us to provide an numpy array
        # we choose to have number of examples on first axis ('b'),
        # then rows and columns of the image, then the channels
        # always 3 in our case
        axes = ('b', 0, 1, 'c')
        dense_x = numpy.empty(shape=(len(ind), image_size, image_size, 3),
                              dtype='uint8')
        categories = []
        has_categ = False
        for i, img in enumerate(ind):
            largest = max(img.shape)
            # do we need to enlarge / shrink the image?
            if largest != image_size:
                wpercent = image_size / float(img.size[0])
                imgin = img.resize((int(img.size[0]*wpercent,hsize),
                                    int(img.size[1]*wpercent,hsize)),
                                   PIL.Image.ANTIALIAS)
            else:
                imgin = img
            # convert to coresponding numpy array
            imgin = numpy.array(imgin)
            dense_x[i] = imgin
            categories.append(ind[img])
            if ind[img] != '':
                has_categ = True

        # if we have categories / values convert them to proper format
        if has_categ:
            if regression:
                # in regression we expect real values
                dense_y = numpy.empty(shape=(len(ind), 1), dtype=floatX)
                for i, ctg in enumerate(categories):
                    dense_y[i, 0] = float(ctg)
            else:
                # in classification we expect integers
                dense_y = numpy.empty(shape=(len(ind), 1), dtype=int)
                for i, ctg in enumerate(categories):
                    dense_y[i, 0] = int(ctg)
        else:
            dense_y = None

        if rng is None:
            rng = DenseDesignMatrix._default_seed 

        # everything else is handled by the DenseDesignMatrix superclass
        super(Images, self).__init(topo_view=result_x,
                                   y=dense_y,
                                   axes=axes,
                                   preprocessor=preprocessor,
                                   fit_preprocessor=fit_preprocessor,
                                   X_labels=None, y_labels=None)

    def _load_csv(self, csv_path):
        """
        Internal method for loading the content from a .csv file.

        Returns
        -------
        result : OrderedDict
        The method creates a dictionary that should be passed to
        `_load_dict()`.
        """

        # we're going to accumulate files and categories here
        result = OrderedDict()

        # compute absolute path of the source csv file
        csv_path = os.path.abspath(csv_path)

        with open(csv_path, 'rt') as fhand:
            # the reader is flexible, allowing delimiters
            # other than comma; quotation can also be customized
            csvr = csv.reader(fhand,
                              delimiter=',',
                              quotechar='"')

            # the reader will give us a list for each row of
            # the source file
            for i, row in enumerate(csvr):
                # we're going to skip empty rows without warning
                if len(row) == 0:
                    continue
                # we could skip the header here, if present; we 
                # could even detect the column index from its
                # name; but we try to keep the things simple

                # class/value is always first, file path second
                result[row[1]] = row[0]

        return result


    def _load_dict(self, srcdict):
        """
        Internal method for loading the content from a dictionary.

        Image files and numpy arrays are converted to `pil.Image`; empty classes
        are normalized to a string of lenghth 0.

        Returns
        -------
        result : dict
        The method creates a dictionary, with keys being `pil.Image` instances
        and values being classes (None for no class).
        """

        # we're going to accumulate Images and categories here
        result = OrderedDict()

        for img in srcdict:
            if isinstance(img, basestring):
                imgin = Image.open(img)
            elif isinstance(img, numpy.ndarray):
                imgin = Image.fromarray(img)
            elif isinstance(img, Image.Image):
                imgin = img
            else:
                raise ValueError("Valid input for images are strings (a "
                                 "path towards a file), pil images and numpy arrays; "
                                 "%s is not supported" % str(img.__class__))
            cls = srcdict[img]
            if cls is None:
                cls = ''
            result[imgin] = cls
        return result













