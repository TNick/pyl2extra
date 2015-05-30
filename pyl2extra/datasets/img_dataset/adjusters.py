"""
Classes that provide adjustings to the ImgDataset.

These preprocess data but the name preprocessor is not used to avoid
confusion with pylearn2 preprocessors.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from copy import copy
import functools
import numpy
import os
import Image
import logging
import webcolors
from scipy.ndimage.interpolation import rotate, zoom
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import serial
import theano

from pyl2extra.utils.paramstore import ParamStore
from pyl2extra.datasets.img_dataset.data_providers import DeDeMaProvider


class Adjuster(object):
    """
    Abstract class providing the interface for adjusters.

    Each Adjuster is associated with a dataset through its ``setup()`` method.
    """
    def __init__(self):
        #: provides its parameters in required order
        self.prmstore = None
        #: the dataset that owns this adjuster (set-up by `setup()`)
        self.dataset = None
        #: iteration mode for parameters in process() method
        self.mode = None
        super(Adjuster, self).__init__()

    def setup(self, dataset, mode):
        """
        The dataset uses this method after it initialized itself.

        Parameters
        ----------
        dataset : ImgDataset
            The dataset in question.
        mode : str
            Iteration mode; valid values are described in
            `pyl2extra.utils.paramstore.ParamStore`.

        Returns
        -------
        categ : str
            A string indicating the category.
        """
        pass
        #assert isinstance(dataset, ImgDataset)

    def tear_down(self):
        """
        Called by the dataset fromits tear_down() method.
        """
        pass

    def transf_count(self):
        """
        Tell the number of images will be generated from a single image.

        Returns
        -------
        count : int
            A non-negative integer.
        """
        return 0

    def process(self, batch):
        """
        The instance is requested to transform the input.

        Parameters
        ----------
        batch : numpy.array
            An array to process. The expected shape is ('b', W, H, 'c'),
            with c being 3 or 4: red, green, blue and (optionally) alpha.
            All channels are expected to provide input in [0;1] interval.

        Returns
        -------
        batch : numpy.array
            The resulted batch, processed. The result's shape is
            ('b', W, H, 'c'), with c being 4: red, green, blue and
            (optionally) alpha. Values for all channels are in the [0;1]
            interval most of the time.
        """
        raise NotImplementedError()

    def accumulate(self, batch):
        """
        The instance is requested to transform the input in all possible ways.

        The size of the output will be input size * transf_count()

        Parameters
        ----------
        batch : numpy.array
            An array to process. The expected shape is ('b', W, H, 'c'),
            with c being 3 or 4: red, green, blue and (optionally) alpha.
            All channels are expected to provide input in [0;1] interval.

        Returns
        -------
        batch : numpy.array
            The resulted batch, processed. The result's shape is
            ('b', W, H, 'c'), with c being 4: red, green, blue and
            (optionally) alpha. Values for all channels are in the [0;1]
            interval most of the time.
        """
        raise NotImplementedError()

    def accum_text(self, size):
        """
        Does the same job as `accumulate()` but with text.
        
        The values for the parameters are added along with parameter name
        in same way as `accumulate()` accumulates them.
        """
        raise NotImplementedError()


class BackgroundAdj(Adjuster):
    """
    Replaces the alpha channel with specified background(s).

    Parameters
    ----------
    backgrounds : tuple, string or iterable, optional
        The backgrounds to apply to the image. It can be an image (that
        will be scaled to desired size), a color represented as a RGB tuple,
        a color represented as a string or a list of any combination of
        the above. The color components must be in the [0;255] interval and
        will be internally converted to [0;1]
    image_files : list of strings, optional
        Paths towards image files to be used as backgrounds.
        It is recomended to use this member to provide images instead of
        providing images directly, as there seems to be a bug in serializing
        Image instances (this is important if inter-process communication
        is required).

    Notes
    -----
    See webcolors module for valid string formats for the ``background``
    member.
    """
    def __init__(self, backgrounds=None, image_files=None):

        #: list of colors and images to use as backgrounds
        self.backgrounds = []
        self.original_backgrounds = backgrounds
        self.original_image_files = image_files

        BackgroundAdj._normalize_back(self.backgrounds,
                                      backgrounds,
                                      image_files)
        assert len(self.backgrounds) > 0
        super(BackgroundAdj, self).__init__()

    @functools.wraps(Adjuster.setup)
    def setup(self, dataset, mode):
        logging.debug('%s is being set-up', self.__class__.__name__)
        #assert isinstance(dataset, ImgDataset)
        self.prmstore = ParamStore([self.backgrounds], mode=mode)
        self.mode = mode

    @functools.wraps(Adjuster.transf_count)
    def transf_count(self):
        return len(self.backgrounds)

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        hash_val = hash(self.__class__.__name__)
        for bkg in self.backgrounds:
            hash_val = hash_val ^ hash(bkg)
        return hash_val

    @staticmethod
    def _normalize_back(result, value, image_files=None):
        """
        Converts user provided list of backgrounds into a list of colors
        and images.
        """
        if ((isinstance(value, tuple) or isinstance(value, list)) and
                len(value) == 3 and
                all([isinstance(i, int) for i in value])):
            # a single color represented as a RBG tuple
            result.append(tuple(i/255.0 for i in value))
        elif isinstance(value, basestring):
            # a single color represented as a string
            if value.startswith('#'):
                value = webcolors.hex_to_rgb(value.lower())
            else:
                value = webcolors.name_to_rgb(value.lower())
            result.append(tuple(i/255.0 for i in value))
        elif value is None:
            # default background is black
            result.append((0., 0., 0.))
        elif isinstance(value, Image.Image):
            # an actual image
            value = numpy.array(value.convert('RGBA')) / 255.0
            result.append(value)
        elif isinstance(value, list) or isinstance(value, tuple):
            # a list of the above
            for bkinst in value:
                BackgroundAdj._normalize_back(result, bkinst)
        else:
            raise ValueError("%s is not a valid value for "
                             "MakeSquareAdj's backgrounds" % str(value))

        # image files
        if not image_files is None:
            if isinstance(image_files, basestring):
                image_files = [image_files]
            for imgf in image_files:
                value = Image.open(imgf).convert('RGBA')
                value = numpy.array(value) / 255.0
                result.append(value)

        return result

    def _proc_init(self, batch, tranc=1):
        """
        Common initialization code for process() and accumulate()
        """
        if batch.shape[3] != 4:
            raise AssertionError("BackgroundAdj expects the input to have "
                                 "four channels (red, green, blue and alpha); "
                                 "the adjuster then replaces the alpha zones "
                                 "with given background; provided shape was "
                                 "%s" % str(batch.shape))
        result = numpy.empty(shape=[batch.shape[0]*tranc,
                                    batch.shape[1],
                                    batch.shape[2],
                                    3],
                             dtype=batch.dtype)
        width = batch.shape[2]
        height = batch.shape[1]
        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)
        return result, width, height
        
    def _proc_bkg(self, bkg, width, height):
        """
        Common code for process() and accumulate()
        """
        if isinstance(bkg, numpy.ndarray):
            if not (height == bkg.shape[0] and width == bkg.shape[1]):
                bkg = zoom(bkg, zoom=(height/bkg.shape[0],
                                      width/bkg.shape[1],
                                      1.),
                           output=None,
                           order=3,
                           mode='constant',
                           cval=0.0, prefilter=True)
            assert bkg.shape[2] == 3
        else:
            bkg = numpy.array(bkg*width*height).reshape(height, width, 3)
        return bkg
        
    def _proc_img(self, img, bkg, width, height):
        """
        Common code transforming the images for process() and accumulate()
        """
        mask = (img[:, :, 3]).reshape((height,
                                       width,
                                       1)).repeat(3, axis=-1)
        # make the mask take values >= 0 and <= 1
        assert mask.min() >= 0.0 and mask.max() <= 1.0
        bkmask = 1.0 - mask
        return (numpy.multiply(mask, img[:, :, 0:3]) +
                numpy.multiply(bkmask, bkg))
                                      
    @functools.wraps(Adjuster.process)
    def process(self, batch):
        result, width, height = self._proc_init(batch)

        # apply the background to each image
        for i in range(batch.shape[0]):
            # get or create the background
            bkg = self._proc_bkg(self.prmstore.next()[0], width, height)
            result[i, :, :, :] =self._proc_img(batch[i, :, :, :],
                                               bkg, width, height)  
        assert numpy.all(result >= 0.0) and numpy.all(result <= 1.0)      
        return result

    @functools.wraps(Adjuster.process)
    def accumulate(self, batch):
        tranc = self.transf_count()
        result, width, height = self._proc_init(batch, tranc)

        j = 0
        for bkg in self.prmstore.parameters[0]:
            bkg = self._proc_bkg(bkg, width, height)
            for i in range(batch.shape[0]):
                result[j, :, :, :] = self._proc_img(batch[i, :, :, :],
                                                    bkg, width, height)                      
                j = j + 1
        assert numpy.all(result >= 0.0) and numpy.all(result <= 1.0)
        return result

    @functools.wraps(Adjuster.process)
    def accum_text(self, batch):
        result = []
        for bkg in self.prmstore.parameters[0]:
            for btc in batch:
                btc = copy(btc)
                btc.append({'background': str(bkg)})
                result.append(btc)
        return result

    def __getstate__(self):
        """
        Help pickle this instance.
        """
        state = {}
        state['backgrounds'] = self.original_backgrounds
        state['image_file'] = self.original_image_files
        state['mode'] = self.mode if hasattr(self, 'mode') else None
        return state

    def __setstate__(self, state):
        """
        Help un-pickle this instance.
        """

        self.mode = state['mode']
        self.original_backgrounds = state['backgrounds']
        self.original_image_files = state['image_file']

        self.backgrounds = []
        BackgroundAdj._normalize_back(self.backgrounds,
                                      self.original_backgrounds,
                                      self.original_image_files)
        self.prmstore = ParamStore([self.backgrounds], mode=self.mode)
        assert len(self.backgrounds) > 0


class MakeSquareAdj(Adjuster):
    """
    Scales the image up or down and moves it at the center of the square.

    This adjuster has no dynamic parameters.

    Parameters
    ----------
    size : int, optional
        The size of the resulted image.
    order : int, optional
        The order of the spline interpolation, default is 3. The order
        has to be in the range 0-5.
    construct : bool, optional
        Profiler shows that this is one of the most expensive opperation and,
        because it adds no parameters can be performed at setup time. If this
        option is enabled the adjuster will request all images from currently
        installed data provider, adjust them, create a DenseDesignMatrix
        for it and sue it as the data provider.
    Notes
    -----
    See webcolors module for valid string formats for the ``background``
    member.
    """
    def __init__(self, size=128, order=3, construct=False, cache_loc=None):

        #: order of spline
        self.order = order
        #: size of the resulted image
        self.size = size
        #: location of cache
        self.cache_loc = cache_loc
        #: preprocess inside setup
        self.construct = construct
        super(MakeSquareAdj, self).__init__()

    @functools.wraps(Adjuster.setup)
    def setup(self, dataset, mode):
        logging.debug('%s is being set-up', self.__class__.__name__)
        # override dataset shape because we know better
        dataset.shape = (self.size, self.size)
        #assert isinstance(dataset, ImgDataset)
        if self.construct:
            ddm = self.create_ddm(dataset.data_provider, self.cache_loc)
            new_prov = DeDeMaProvider(ddm)
            new_prov.setup(dataset)
            dataset.data_provider = new_prov

    def create_ddm(self, datap, cache_loc=None):
        """
        Creates a DenseDesignMatrix from a data provider.

        Parameters
        ----------
        datap : pyl2extra.datasets.img_dataset.data_providers.Provider
            The provider to use.
        cache_loc : str, optional
            If provided, must be a path towards a pickled dataset. If the
            file exists it is loaded and returned; otherwise new dataset
            is saved to that location.

        Returns
        -------
        ddm : pylearn2.datasets.dense_design_matrix.DenseDesignMatrix
            The new dataset.
        """
        if cache_loc and os.path.isfile(cache_loc):
            logging.debug('create_ddm reads the data from cache (%s)',
                          cache_loc)
            ddm = serial.load(cache_loc)
            return ddm

        logging.debug('create_ddm creates new dataset')
        # number of examples
        excnt = len(datap)
        assert excnt > 0
        # this will hold all examples in topological order
        examples = numpy.empty(shape=(excnt, self.size, self.size, 4),
                               dtype=theano.config.floatX)
        categs = range(excnt)
        for i, f_path in enumerate(datap):
            # returned image always has 4 channels
            imarray, categs[i] = datap.read(f_path)
            width = imarray.shape[1]
            height = imarray.shape[0]
            assert numpy.all(imarray >= 0.0) and numpy.all(imarray <= 1.0)

            if width == height:
                square = imarray
                largest = width
            else:
                largest = max(width, height)
                deltax = (largest - width) / 2
                deltay = (largest - height) / 2
                # by having zeros the areas not covered by the image
                # become transparent
                square = numpy.zeros(shape=(largest, largest,
                                            imarray.shape[2]),
                                     dtype=theano.config.floatX)
                square[deltay:deltay+height, deltax:deltax+width, :] = imarray

            if self.size == largest:
                examples[i, :, :, :] = square
            else:
                factor = self.size / float(largest)
                zoom(square, zoom=(factor, factor, 1.),
                     output=examples[i, :, :, :],
                     order=self.order,
                     mode='constant',
                     cval=0.0, prefilter=True)
        assert numpy.all(examples >= 0.0) and numpy.all(examples <= 1.0)

        ddm = DenseDesignMatrix(topo_view=examples, y=numpy.array(categs))
        logging.debug('create_ddm has created the new dataset')
        if cache_loc:
            logging.debug('create_ddm caches new dataset at %s', cache_loc)
            ddm.use_design_loc(cache_loc + '.npy')
            serial.save(cache_loc, ddm)
        return ddm

    @functools.wraps(Adjuster.transf_count)
    def transf_count(self):
        return 1

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        hash_val = hash(self.__class__.__name__) ^ hash(self.size)
        return hash_val

    @functools.wraps(Adjuster.process)
    def process(self, batch):
        if batch.shape[3] != 3 and batch.shape[3] != 4:
            raise AssertionError("MakeSquareAdj expects the input to have "
                                 "three or four channels (red, "
                                 "green, blue and alpha); provided shape was "
                                 "%s" % str(batch.shape))

        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)        
        
        # make it square
        width = batch.shape[2]
        height = batch.shape[1]
        largest = max(width, height)
        deltax = (largest - width) / 2
        deltay = (largest - height) / 2
        square = numpy.zeros(shape=(batch.shape[0],
                                    largest, largest,
                                    batch.shape[3]),
                             dtype=batch.dtype)
        square[:, deltay:deltay+height, deltax:deltax+width, :] = batch
        # resize it
        if self.size != largest:
            factor = self.size / float(largest)
            square = zoom(square, zoom=(1., factor, factor, 1.),
                          output=None, order=self.order, mode='constant',
                          cval=0.0, prefilter=True)
            # is it better to re-normalize or to cut values outside [0;1]
            square = (square - square.min()) / (square.max() - square.min())
        assert numpy.all(square >= 0.0) and numpy.all(square <= 1.0) 
        return square

    @functools.wraps(Adjuster.process)
    def accumulate(self, batch):
        return self.process(batch)

    @functools.wraps(Adjuster.process)
    def accum_text(self, batch):
        result = []
        for btc in batch:
            btc = copy(btc)
            btc.append({'squared': str(self.size)})
            result.append(btc)
        return result

class FlipAdj(Adjuster):
    """
    Allows the image to be flipped along x and/or y axis.

    Parameters
    ----------
    horizontal : bool
        The image is to be flipped along the y (vertical) axis.
    vertical : bool
        The image is to be flipped along the x (horizontal) axis.
    """
    def __init__(self, horizontal=True, vertical=False):
        assert horizontal or vertical

        #: horizontal flip enabled or disabled
        self.horizontal = horizontal

        #: vertical flip enabled or disabled
        self.vertical = vertical

        super(FlipAdj, self).__init__()

    @functools.wraps(Adjuster.setup)
    def setup(self, dataset, mode):
        logging.debug('%s is being set-up', self.__class__.__name__)
        #assert isinstance(dataset, ImgDataset)
        lst = [[True, False]]
        if self.horizontal and self.vertical:
            lst.append([True, False])
        self.prmstore = ParamStore(lst, mode=mode)

    @functools.wraps(Adjuster.transf_count)
    def transf_count(self):
        if self.horizontal and self.vertical:
            return 4
        else:
            assert self.horizontal or self.vertical
            return 2

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        return hash(self.__class__.__name__) ^ (hash(self.horizontal) << 1) ^ \
            hash(self.vertical)

    def _get_params(self):
        """
        Retreive the parameters and normalize arrangement.
        """
        flip_horiz = False
        flip_vert = False

        prm = self.prmstore.next()
        if self.horizontal and self.vertical:
            flip_horiz = prm[0]
            flip_vert = prm[1]
        elif self.horizontal:
            flip_horiz = prm[0]
        elif self.vertical:
            flip_vert = prm[0]
        else:
            raise ValueError("Either horizontal or vertical "
                             "should be selected for FlipAdj")
        return flip_horiz, flip_vert

    @functools.wraps(Adjuster.process)
    def process(self, batch):
        for i in range(batch.shape[0]):
            img = batch[i, :, :, :]
            # get parameters for this image
            flip_horiz, flip_vert = self._get_params()
            if flip_horiz:
                img = img[:, ::-1, :]
            if flip_vert:
                img = img[::-1, :, :]
            batch[i, :, :, :] = img
        return batch

    @functools.wraps(Adjuster.process)
    def accumulate(self, batch):
        tranc = self.transf_count()
        result = numpy.empty(shape=[batch.shape[0]*tranc,
                                    batch.shape[1],
                                    batch.shape[2],
                                    batch.shape[3]],
                             dtype=batch.dtype)
        # original image
        off = 0
        count = batch.shape[0]
        result[off:count, :, :, :] = batch[:, :, :, :]

        off = batch.shape[0]
        if self.vertical:
            result[off:off+count, :, :, :] = batch[:, ::-1, :, :]
            count = count * 2
            off = batch.shape[0] * 2
        # flip both original and vflipped image
        if self.horizontal:
            result[off:off+count, :, :, :] = result[0:off, :, ::-1, :]
        return result

    @functools.wraps(Adjuster.process)
    def accum_text(self, batch):
        result = []
        for btc in batch:
            btc = copy(btc)
            btc.append({'flipped': 'not'})
            result.append(btc)
        if self.vertical:
            for btc in batch:
                btc = copy(btc)
                btc.append({'flipped': 'vertically'})
                result.append(btc)
        if self.horizontal:
            for btc in batch:
                btc = copy(btc)
                btc.append({'flipped': 'horizontally'})
                result.append(btc)
        if self.vertical and self.horizontal:
            for btc in batch:
                btc = copy(btc)
                btc.append({'flipped': 'both'})
                result.append(btc)
        return result


class RotationAdj(Adjuster):
    """
    Allows the image to be rotated in the x-y plane.

    Example: ``min_deg=-45.0, max_deg=45.0, step=15.0``
    Folowing angles are generated: -45.0, -30.0, -15.0, 0.0, 15.0, 30.0,  45.0

    Parameters
    ----------
    min_deg : float, optional
        The angle (degrees) where the iteration starts.
        First image will have exactly this angle of rotation.
    max_deg : float, optional
        Maximum angle (degrees); the iteration will stop before or exactly at
        this angle, based on the ``min_deg`` and ``step``.
    step : float, optional
        The rotation angle is incremented each time with this ammount
        (degrees). If this parameter is 0 a single image will be generated,
        rotated by ``min_deg`` degrees.
    order : int, optional
        The order of the spline interpolation, default is 3. The order
        has to be in the range 0-5.
    resize : bool, optional
        If True the image is resized, then the bounding box of the new image
        is returned (will have dark corners); an additional opperation is
        needed to scale the image back to original size. If False the image
        is rotated in place, resulting in corners being cut. By default
        `resize` is False because is faster.
    """
    def __init__(self, min_deg=-45.0, max_deg=45.0, step=15.0,
                 order=3, resize=False):

        #: first angle to use
        self.min_deg = min_deg

        #: limit angle
        self.max_deg = max_deg

        #: the step to use
        self.step = step
        assert step >= 0

        if min_deg > max_deg:
            self.min_deg = max_deg
            self.max_deg = min_deg

        #: order of spline interpolation
        self.order = order
        assert order >= 0 and order <= 5

        #: resize or not
        self.resize = resize

        super(RotationAdj, self).__init__()

    @functools.wraps(Adjuster.setup)
    def setup(self, dataset, mode):
        logging.debug('%s is being set-up', self.__class__.__name__)
        #assert isinstance(dataset, ImgDataset)
        angles = []
        angle = self.min_deg
        if self.step == 0.0:
            angles.append(angle)
        else:
            while angle <= self.max_deg:
                angles.append(angle)
                angle = angle + self.step
        self.prmstore = ParamStore([angles], mode=mode)

    @functools.wraps(Adjuster.transf_count)
    def transf_count(self):
        if self.step == 0.0:
            return 1
        else:
            return numpy.floor((self.max_deg - self.min_deg) / self.step) + 1

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        return hash(self.__class__.__name__) ^ (hash(self.step) << 2) ^ \
            (hash(self.max_deg) << 1) ^ hash(self.min_deg) ^ \
            (self.order ^ 3) ^ ((self.resize * 5) ^ 4)

    @functools.wraps(Adjuster.process)
    def process(self, batch):
        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)
        if batch.shape[3] != 3 and batch.shape[3] != 4:
            raise AssertionError("RotationAdj expects the input to have "
                                 "three or four channels (red, "
                                 "green, blue and alpha); provided shape was "
                                 "%s" % str(batch.shape))

        for i in range(batch.shape[0]):
            img = batch[i, :, :, :]
            # get parameters for this image
            angle = self.prmstore.next()[0]
            if self.resize:
                img = rotate(img, angle, axes=(0, 1), reshape=True,
                             output=None, order=self.order,
                             mode='constant', cval=0.0, prefilter=True)
                zoom(img, zoom=(float(batch.shape[1]) / float(img.shape[0]),
                                float(batch.shape[2]) / float(img.shape[1]),
                                1.), output=batch[i, :, :, :],
                     order=self.order, mode='constant',
                     cval=0.0, prefilter=True)
            else:
                rotate(img, angle, axes=(0, 1), reshape=False,
                       output=img, order=self.order,
                       mode='constant', cval=0.0, prefilter=True)
        batch = (batch - batch.min()) / (batch.max() - batch.min())
        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)
        return batch

    @functools.wraps(Adjuster.process)
    def accumulate(self, batch):
        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)
        tranc = self.transf_count()
        result = numpy.empty(shape=[batch.shape[0]*tranc,
                                    batch.shape[1],
                                    batch.shape[2],
                                    batch.shape[3]],
                             dtype=batch.dtype)

        off = 0
        oend = batch.shape[0]
        for angle in self.prmstore.parameters[0]:
            if self.resize:
                img = rotate(batch, angle, axes=(1, 2), reshape=True,
                             output=None, order=self.order,
                             mode='constant', cval=0.0, prefilter=True)
                zoom(img, zoom=(1.,
                                float(batch.shape[1]) / float(img.shape[1]),
                                float(batch.shape[2]) / float(img.shape[2]),
                                1.), output=result[off:oend, :, :, :],
                     order=self.order, mode='constant',
                     cval=0.0, prefilter=True)
            else:
                rotate(batch, angle, axes=(1, 2), reshape=False,
                       output=result[off:oend, :, :, :], order=self.order,
                       mode='constant', cval=0.0, prefilter=True)
            off = off + batch.shape[0]
            oend = oend + batch.shape[0]
            
        # is it better to re-normalize or to cut values outside [0;1]
        result = (result - result.min()) / (result.max() - result.min())
        assert numpy.all(result >= 0.0) and numpy.all(result <= 1.0)
        return result

    @functools.wraps(Adjuster.process)
    def accum_text(self, batch):
        result = []
        for angle in self.prmstore.parameters[0]:
            for btc in batch:
                btc = copy(btc)
                btc.append({'angle': str(angle) + ' deg'})
                result.append(btc)
        return result
        
        
class ScalePatchAdj(Adjuster):
    """
    Scales the image and places the result in the four corners of the image
    and in center.

    Parameters
    ----------
    outsize: tuple, optional
        Tuple of size 2 indicating the output shape; if ``None`` (default) the
        shape of the input batch is preserved.
    start_factor : float, optional
        The scale factor where the iteration starts.
        First image will have exactly this scale.
    end_factor : float, optional
        Maximum scale factor; the iteration will stop before or exactly at
        this factor, based on the ``start_factor`` and ``step``.
    step : float, optional
        The factor is incremented each time with this ammount.
        If this parameter is 0 five images will be generated,
        all scaled by ``start_factor``.
    placements : list, optional
        List of places where the image should be moved after scaling;
        members can be strings ('top_left', 'top_right', 'btm_left',
        'btm_right', 'center') or callable objects.
        The callable receives the shape of the image and the ``outsize``
        and must return the placement as a tuple (delta_x, delta_y)
    order : int, optional
        The order of the spline interpolation, default is 3. The order
        has to be in the range 0-5.

    Notes
    -----
    The indices used may be confusing. For the batch
    - 0: all images (batch.shape[0] == number of images)
    - 1: all rows in an image (batch.shape[1] == height)
    - 2: all columns in a row (batch.shape[2] == width)
    - 3: all channels for a pixel (batch.shape[3] == number of channels)

    For ``outsize`` parameter and attribute ``outsize[0] == width``,
    ``outsize[1] == height``.

    The image that is provided as first argument to callables in
    ``placements`` don't have the batch dimension, so:
    - 0: all rows in the image (batch.shape[0] == height)
    - 1: all columns in a row (batch.shape[1] == width)
    - 2: all channels for a pixel (batch.shape[2] == number of channels)
    """
    def __init__(self, outsize=None,
                 start_factor=0.8, end_factor=0.9, step=0.1,
                 placements=None, order=3):
        #: output shape for the image (width, height)
        self.outsize = outsize
        if not outsize is None:
            assert len(outsize) == 2

        #: first angle to use
        self.start_factor = start_factor
        #: limit angle
        self.end_factor = end_factor
        #: the step to use
        self.step = step
        assert step >= 0
        if start_factor > end_factor:
            self.start_factor = end_factor
            self.end_factor = start_factor

        if placements is None:
            placements = ScalePatchAdj.default_placements
        else:
            for plcm in placements:
                if isinstance(plcm, basestring):
                    assert plcm in ScalePatchAdj.default_placements
                else:
                    assert hasattr(plcm, '__call__')
        #: list of places where to move the image
        self.placements = placements

        #: order of spline interpolation
        self.order = order
        assert order >= 0 and order <= 5

        super(ScalePatchAdj, self).__init__()

    default_placements = ['top_left', 'top_right',
                          'btm_left', 'btm_right',
                          'center']

    @staticmethod
    def top_left(img_shape, outsize):
        """Places the image in the top left corner."""
        return (0, 0)

    @staticmethod
    def top_right(img_shape, outsize):
        """Places the image in the top right corner."""
        return (outsize[0] - img_shape[1], 0)

    @staticmethod
    def btm_left(img_shape, outsize):
        """Places the image in the bottom left corner."""
        return (0, outsize[1] - img_shape[0])

    @staticmethod
    def btm_right(img_shape, outsize):
        """Places the image in the bottom right corner."""
        return (outsize[0] - img_shape[1], outsize[1] - img_shape[0])

    @staticmethod
    def center(img_shape, outsize):
        """Places the image in the center."""
        return ((outsize[0] - img_shape[1])/2,
                (outsize[1] - img_shape[0])/2)

    @functools.wraps(Adjuster.setup)
    def setup(self, dataset, mode):
        logging.debug('%s is being set-up', self.__class__.__name__)
        #assert isinstance(dataset, ImgDataset)
        pos = []
        for plcm in self.placements:
            if isinstance(plcm, basestring):
                if plcm == 'top_left':
                    plcm = ScalePatchAdj.top_left
                elif plcm == 'top_right':
                    plcm = ScalePatchAdj.top_right
                elif plcm == 'btm_left':
                    plcm = ScalePatchAdj.btm_left
                elif plcm == 'btm_right':
                    plcm = ScalePatchAdj.btm_right
                elif plcm == 'center':
                    plcm = ScalePatchAdj.center
                else:
                    raise ValueError('Unexpected placement name: %s' % plcm)
            pos.append(plcm)

        factors = []
        factor = self.start_factor
        while factor <= self.end_factor:
            factors.append(factor)
            factor = factor + self.step
        self.prmstore = ParamStore([pos, factors], mode=mode)

    @functools.wraps(Adjuster.transf_count)
    def transf_count(self):
        count = len(self.placements)
        if self.step == 0.0:
            return count
        else:
            return count * (numpy.floor(1.0e-8 +
                                        (self.end_factor -
                                         self.start_factor) /
                                        self.step) + 1)

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        return hash(self.__class__.__name__) ^ (hash(self.step) << 2) ^ \
            (hash(self.end_factor) << 1) ^ hash(self.start_factor) ^ \
            (self.order << 3) ^ (hash(self.end_factor) << 4)

    def _process_img(self, img, outsize, scaled_shape,
                     result, placement, factor, j):
        """
        """
        assert outsize[0] > scaled_shape[1]
        assert outsize[1] > scaled_shape[0]
        deltax, deltay = placement(scaled_shape, outsize)
        assert deltax >= 0 and deltay >= 0
        endx = deltax + scaled_shape[1]
        endy = deltay + scaled_shape[0]
        assert endx <= outsize[0] and endy <= outsize[1]
        zoom(img, zoom=(factor, factor, 1.),
             output=result[j, deltay:endy, deltax:endx, :],
             order=self.order, mode='constant',
             cval=0.0, prefilter=True)
        if deltay > 0:
            result[j, 0:deltay, :, :].fill(0.)
        if deltax > 0:
            result[j, :, 0:deltax, :].fill(0.)
        if endy < outsize[1]:
            result[j, endy:outsize[1], :, :].fill(0.)
        if endx < outsize[0]:
            result[j, :, endx:outsize[0], :].fill(0.)

    @functools.wraps(Adjuster.process)
    def process(self, batch):
        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)
        if batch.shape[3] != 3 and batch.shape[3] != 4:
            raise AssertionError("ScalePatchAdj expects the input to have "
                                 "three or four channels (red, "
                                 "green, blue and alpha); provided shape was "
                                 "%s" % str(batch.shape))
        if ((self.outsize is None) or
                ((self.outsize[0] == batch.shape[2]) and
                 (self.outsize[1] == batch.shape[1]))):
            result = batch
            outsize = (batch.shape[2], batch.shape[1])
        else:
            result = numpy.empty(shape=(batch.shape[0],
                                        self.outsize[1],
                                        self.outsize[0],
                                        batch.shape[3]),
                                 dtype=batch.dtype)
            outsize = self.outsize

        for i in range(batch.shape[0]):
            # get parameters for this image
            placement, factor = self.prmstore.next()
            scaled_shape = tuple(
                [int(round(ii * jj)) for ii, jj in zip(batch.shape[1:3],
                                                       (factor, factor))])
            self._process_img(batch[i, :, :, :],
                              outsize, scaled_shape,
                              result, placement, factor, i)

        result = (result - result.min()) / (result.max() - result.min())
        assert numpy.all(result >= 0.0) and numpy.all(result <= 1.0)
        return result

    @functools.wraps(Adjuster.process)
    def accumulate(self, batch):
        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)
        tranc = self.transf_count()
        if self.outsize is None:
            outsize = (batch.shape[2], batch.shape[1])
        else:
            outsize = self.outsize

        result = numpy.empty(shape=(batch.shape[0]*tranc,
                                    outsize[1],
                                    outsize[0],
                                    batch.shape[3]),
                             dtype=batch.dtype)
        j = 0
        for placement in self.prmstore.parameters[0]:
            for factor in self.prmstore.parameters[1]:
                scaled_shape = tuple(
                    [int(round(ii * jj)) for ii, jj in zip(batch.shape[1:3],
                                                           (factor, factor))])
                for i in range(batch.shape[0]):
                    self._process_img(batch[i, :, :, :],
                                      outsize, scaled_shape,
                                      result, placement, factor, j)
                    j = j + 1
                    
        # is it better to re-normalize or to cut values outside [0;1]
        result = (result - result.min()) / (result.max() - result.min())
        assert numpy.all(result >= 0.0) and numpy.all(result <= 1.0)
        return result

    @staticmethod
    def place_to_str(func):
        """
        Convert a placement function to a string.
        """
        if func == ScalePatchAdj.top_left:
            return 'top left'
        elif func == ScalePatchAdj.top_right:
            return 'top left'
        elif func == ScalePatchAdj.btm_left:
            return 'top left'
        elif func == ScalePatchAdj.btm_right:
            return 'top left'
        elif func == ScalePatchAdj.center:
            return 'top left'
        else:
            return 'custom'
        
    @functools.wraps(Adjuster.process)
    def accum_text(self, batch):
        result = []
        for placement in self.prmstore.parameters[0]:
            for factor in self.prmstore.parameters[1]:
                for btc in batch:
                    btc = copy(btc)
                    plc = ScalePatchAdj.place_to_str(placement)
                    btc.append({'placement': plc,
                                'factor': str(factor)})
                    result.append(btc)
        return result


class GcaAdj(Adjuster):
    """
    Applies global contrast normalization channel-wise.

    See ``global_contrast_normalize`` in ``pylearn2.expr.preprocessing`` for
    more information.

    Parameters
    ----------
    start_scale : float, optional
        Multiply features by this const.
        The scale factor where the iteration starts.
        First image will have exactly this scale.
    end_scale : float, optional
        Maximum scale factor; the iteration will stop before or exactly at
        this factor, based on the ``start_factor`` and ``step_scale``.
    step_scale : float, optional
        The factor is incremented each time with this ammount.
        If this parameter is 0 all images will be
        scaled by ``start_factor``.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing.
        If a ``True`` or ``False`` value is provided that value is used for
        all images. If ``None`` both ``True`` and ``False`` are added
        to the list of valid parameters.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm.
        If a ``True`` or ``False`` value is provided that value is used for
        all images. If ``None`` both ``True`` and ``False`` are added
        to the list of valid parameters.
    start_sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
        The bias where the iteration starts.
        First image will have exactly this scale.
    end_sqrt_bias : float, optional
        Maximum bias; the iteration will stop before or exactly at
        this factor, based on the ``start_sqrt_bias`` and ``step_sqrt_bias``.
    step_sqrt_bias : float, optional
        The bias is incremented each time with this ammount.
        If this parameter is 0 all images will be
        biased by ``start_sqrt_bias``.
    """
    def __init__(self, start_scale=1., end_scale=1., step_scale=0.,
                 subtract_mean=None, use_std=None,
                 start_sqrt_bias=0., end_sqrt_bias=0., step_sqrt_bias=0.):

        #: first scale to use
        self.start_scale = start_scale
        #: limit scale
        self.end_scale = end_scale
        #: the step to use for scale
        self.step_scale = step_scale
        assert step_scale >= 0
        if start_scale > end_scale:
            self.start_scale = end_scale
            self.end_scale = start_scale

        if subtract_mean is None:
            subtract_mean = (True, False)
        elif isinstance(subtract_mean, bool):
            subtract_mean = (subtract_mean,)
        else:
            subtract_mean = tuple(subtract_mean)
        #: substract the mean or not or both
        self.subtract_mean = subtract_mean
        if use_std is None:
            use_std = (True, False)
        elif isinstance(use_std, bool):
            use_std = (use_std,)
        else:
            use_std = tuple(use_std)
        #: Normalize by the per-example std-dev, vector norm or both
        self.use_std = use_std

        #: first bias to use
        self.start_sqrt_bias = start_sqrt_bias
        #: limit bias
        self.end_sqrt_bias = end_sqrt_bias
        #: the step to use for bias
        self.step_sqrt_bias = step_sqrt_bias
        assert step_scale >= 0
        if start_sqrt_bias > end_sqrt_bias:
            self.start_sqrt_bias = end_sqrt_bias
            self.end_sqrt_bias = start_sqrt_bias

        super(GcaAdj, self).__init__()

    @functools.wraps(Adjuster.setup)
    def setup(self, dataset, mode):
        logging.debug('%s is being set-up', self.__class__.__name__)
        #assert isinstance(dataset, ImgDataset)
        scales = []
        scale = self.start_scale
        if self.step_scale == 0.0:
            scales.append(scale)
        else:
            while scale <= self.end_scale:
                scales.append(scale)
                scale = scale + self.step_scale
        scales = tuple(scales)

        biases = []
        bias = self.start_sqrt_bias
        if self.step_sqrt_bias == 0.0:
            biases.append(bias)
        else:
            while bias <= self.end_sqrt_bias:
                biases.append(bias)
                bias = bias + self.step_sqrt_bias
        biases = tuple(biases)

        if self.subtract_mean is None:
            subtract_mean = (True, False)
        else:
            subtract_mean = tuple(self.subtract_mean)
        if self.use_std is None:
            use_std = (True, False)
        else:
            use_std = tuple(self.use_std)
        self.prmstore = ParamStore([scales, subtract_mean, use_std, biases],
                                   mode=mode)

    @functools.wraps(Adjuster.transf_count)
    def transf_count(self):
        count = 1
        if self.step_scale != 0.0:
            count = count * (numpy.floor((self.end_scale -
                                          self.start_scale) /
                                         self.step_scale) + 1)
        if self.step_sqrt_bias != 0.0:
            count = count * (numpy.floor((self.end_sqrt_bias -
                                          self.start_sqrt_bias) /
                                         self.step_sqrt_bias) + 1)
        count = count * len(self.subtract_mean)
        count = count * len(self.use_std)

        return count

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        return hash(self.__class__.__name__) ^ \
            (hash(self.start_scale) << 0) ^ \
            (hash(self.end_scale) << 1) ^ \
            (hash(self.step_scale) << 2)  ^ \
            (hash(self.start_sqrt_bias) << 3) ^ \
            (hash(self.end_sqrt_bias) << 4) ^ \
            (hash(self.step_sqrt_bias) << 5)  ^ \
            (hash(self.subtract_mean) << 6) ^ \
            (hash(self.use_std) << 7)

    @functools.wraps(Adjuster.process)
    def process(self, batch):
        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)
        if batch.shape[3] != 3 and batch.shape[3] != 4:
            raise AssertionError("GcaAdj expects the input to have "
                                 "three or four channels (red, "
                                 "green, blue and alpha); provided shape was "
                                 "%s" % str(batch.shape))

        # the image may have an alpha channel that is not normalized
        chan_cnt = min(batch.shape[3], 3)
        chan_sz = batch.shape[1] * batch.shape[2]
        for i in range(batch.shape[0]):
            
            img = batch[i, :, :, :]
            # get parameters for this image
            scale, subtract_mean, use_std, bias = self.prmstore.next()
            gcnarray = img[:, :, 0:3].swapaxes(0, 2).reshape(
                chan_cnt, chan_sz)
            gcnarray = global_contrast_normalize(gcnarray,
                                                 scale=scale,
                                                 subtract_mean=subtract_mean,
                                                 use_std=use_std,
                                                 sqrt_bias=bias,
                                                 min_divisor=1e-8)
            gcnarray = gcnarray.reshape((chan_cnt,
                                         batch.shape[2],
                                         batch.shape[1]))
            img[:, :, 0:3] = gcnarray.swapaxes(0, 2)
        return batch

    @functools.wraps(Adjuster.accumulate)
    def accumulate(self, batch):
        assert numpy.all(batch >= 0.0) and numpy.all(batch <= 1.0)
        if batch.shape[3] != 3 and batch.shape[3] != 4:
            raise AssertionError("GcaAdj expects the input to have "
                                 "three or four channels (red, "
                                 "green, blue and alpha); provided shape was "
                                 "%s" % str(batch.shape))
        chan_cnt = min(batch.shape[3], 3)
        chan_sz = batch.shape[1] * batch.shape[2]
        if batch.shape[3] == 4:
            mask = batch[:, :, :, 3]

        tranc = self.transf_count()
        result = numpy.empty(shape=[batch.shape[0]*tranc,
                                    batch.shape[1],
                                    batch.shape[2],
                                    batch.shape[3]],
                             dtype=batch.dtype)

        j = 0
        for scale in self.prmstore.parameters[0]:
            for subtract_mean in self.prmstore.parameters[1]:
                for use_std in self.prmstore.parameters[2]:
                    for bias in self.prmstore.parameters[3]:
                        
                        if batch.shape[3] == 4:
                            oend = j + batch.shape[0]
                            result[j:oend, :, :, 3] = mask
                        for i in range(batch.shape[0]):
                            img = batch[i, :, :, :]
                            gcnarray = img[:, :, 0:3].swapaxes(0, 2).reshape(
                                chan_cnt, chan_sz)
                            gcnarray = global_contrast_normalize(
                                gcnarray,
                                scale=scale,
                                subtract_mean=subtract_mean,
                                use_std=use_std,
                                sqrt_bias=bias,
                                min_divisor=1e-8)
                            gcnarray = gcnarray.reshape((chan_cnt,
                                                         batch.shape[2],
                                                         batch.shape[1]))
                            result[j, :, :, 0:3] = gcnarray.swapaxes(0, 2)
                            j = j + 1

        # the output is NOT normalized in [0;1]
        return result
        
    @functools.wraps(Adjuster.accum_text)
    def accum_text(self, batch):
        result = []
        for scale in self.prmstore.parameters[0]:
            for subtract_mean in self.prmstore.parameters[1]:
                for use_std in self.prmstore.parameters[2]:
                    for bias in self.prmstore.parameters[3]:
                        for btc in batch:
                            btc = copy(btc)
                            btc.append({'scale': str(scale),
                                        'subtract_mean': str(subtract_mean),
                                        'use_std': str(use_std),
                                        'bias': str(bias)})
                            result.append(btc)
        return result

def tt(xy):
    print xy.min(), xy.max(), xy.shape, xy.dtype
    
def adj_from_string(adj_name):
    """
    Creates an adjuster based on a string key.

    Parameters
    ----------
    adj_name : str
        A string identifying the type of Adjuster to use.

    Returns
    -------
    adj : Adjuster
        The instance that was constructed.
    """

    if adj_name == 'rotate':
        return RotationAdj()
    elif adj_name == 'flip':
        return FlipAdj()
    elif adj_name == 'scale':
        return ScalePatchAdj()
    elif adj_name == 'gca':
        return GcaAdj()
    elif adj_name == 'back':
        return BackgroundAdj()
    else:
        raise ValueError('%s is not a known Adjuster name' % adj_name)
