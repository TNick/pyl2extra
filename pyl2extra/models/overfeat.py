# -*- coding: utf-8 -*-
"""
Layers and models based on OverFeat_ weights.

OverFeat_ is a Convolutional Network-based image classifier
and feature extractor. OverFeat is Copyright NYU 2013. Authors are
Michael Mathieu, Pierre Sermanet, and Yann LeCun. The weights archive
is provided separatelly (here <http://cilvr.cs.nyu.edu/lib/exe/fetch.php?media=overfeat:overfeat-weights.tgz>)_

..code:

    wget http://cilvr.cs.nyu.edu/lib/exe/fetch.php?media=overfeat:overfeat-weights.tgz -O weights.tgz
    tar -xzf weights.tgz

The two files of interest for us are ``net_weight_0`` for small network
and ``net_weight_1`` for large network.

The code in this file was inspired and, in some cases, copied from
(sklearn-theano <https://github.com/sklearn-theano/sklearn-theano>)_ that
is licensed under a BSD 3-clause license by Kyle Kastner and
Michael Eickenberg. OverFeat specifics are mostly in (overfeat.py
<https://github.com/sklearn-theano/sklearn-theano/blob/master/sklearn_theano/feature_extraction/overfeat.py>)_


_OverFeat: https://github.com/sermanet/OverFeat
"""

from PIL import Image
import os
import numbers
import numpy
import time
import theano
from pylearn2.models.model import Model
from pylearn2.datasets.dataset import Dataset
from pylearn2.models.mlp import (MLP, ConvElemwise,
                                 ConvRectifiedLinear, IdentityConvNonlinearity,
                                 Softmax, Layer)
from pylearn2.space import (Conv2DSpace, VectorSpace)
from pylearn2.utils import wraps, safe_zip
from pylearn2.datasets.preprocessing import Preprocessor
from theano import tensor

from pyl2extra.datasets.images import Images
from pyl2extra.models.overfeat_class_labels import get_overfeat_class_label

# architecture dependent representation of real numbers
floatX = theano.config.floatX

# the size of the input image for the two networks
SMALL_INPUT = 231
LARGE_INPUT = 221

# names of the weights files
SMALL_NETWORK_WEIGHT_FILE = 'net_weight_0'
LARGE_NETWORK_WEIGHT_FILE = 'net_weight_1'

# The shape of the network organized as follows
# - number of output channels
# - number of input channels
# - number of rows
# - number of columns
SMALL_NETWORK_FILTER_SHAPES = numpy.array([(96, 3, 11, 11),
                                           (256, 96, 5, 5),
                                           (512, 256, 3, 3),
                                           (1024, 512, 3, 3),
                                           (1024, 1024, 3, 3),
                                           (3072, 1024, 6, 6),
                                           (4096, 3072, 1, 1),
                                           (1000, 4096, 1, 1)])
SMALL_NETWORK_BIAS_SHAPES = SMALL_NETWORK_FILTER_SHAPES[:, 0]

LARGE_NETWORK_FILTER_SHAPES = numpy.array([(96, 3, 7, 7),
                                           (256, 96, 7, 7),
                                           (512, 256, 3, 3),
                                           (512, 512, 3, 3),
                                           (1024, 512, 3, 3),
                                           (1024, 1024, 3, 3),
                                           (4096, 1024, 5, 5),
                                           (4096, 4096, 1, 1),
                                           (1000, 4096, 1, 1)])
LARGE_NETWORK_BIAS_SHAPES = LARGE_NETWORK_FILTER_SHAPES[:, 0]

# the name of the environment variable that is expected to hold the
# path to directory containing network parameters (net_weight_0 and
# net_weight_1)
FILE_PATH_KEY = 'PYL2X_OVERFEAT_PATH'

# parameters for image normalization
STANDARDIZE_MEAN = 118.380948
STANDARDIZE_STD = 61.896913


class Params(object):
    """
    Container for OverFeat weights and biases.

    The constructor simply looiks up the file, reads it and loads
    ``weights`` and ``biases`` attributes with their respective values.

    The set of coresponding pylearn2 layers can be created using
    ``layers()`` method. The weights are NOT copied inside them, allowing
    the user to copy the architecture only.

    To create a full pylearn2 model with all the layers and all the weights
    initialized use ``model()`` method.

    Parameters
    ----------
    large : bool, optional
        Which variant to load (large or - by default - small variant).
    weights_file : str, optional
        The path towards the file to load. By default a file is searched
        in current directory. The name depends on the ``large`` parameter.
    """
    def __init__(self, large=False, weights_file=None):
        if weights_file is None:
            if large:
                weights_file = LARGE_NETWORK_WEIGHT_FILE
            else:
                weights_file = SMALL_NETWORK_WEIGHT_FILE
        while not os.path.isfile(weights_file):
            if not os.path.isabs(weights_file):
                if os.environ.has_key(FILE_PATH_KEY):
                    weights_file = os.path.abspath(
                        os.path.join(os.environ[FILE_PATH_KEY], weights_file))
                else:
                    raise ValueError('Input file %s does not exist; '
                                     'a relative path was provided but %s '
                                     'is not set'
                                     % (weights_file, FILE_PATH_KEY))
            else:
                raise ValueError('Input file %s does not exist' % weights_file)

        #: the variant we're representing
        self.large = large
        #: the file where the parameters were loaded from
        self.weights_file = weights_file

        # read the parameters from file
        memmap = numpy.memmap(weights_file, dtype=numpy.float32)
        mempointer = 0
        weights = []
        biases = []
        for weight_shape, bias_shape in zip(self.filter_shapes,
                                            self.biases_shapes):
            filter_size = numpy.prod(weight_shape)
            wflat = memmap[mempointer:mempointer + filter_size]
            # TODO: i don't really understand why we flip those
            # https://github.com/sklearn-theano/sklearn-theano/blob/master/sklearn_theano/feature_extraction/overfeat.py#L95
            weight = wflat.reshape(weight_shape)[:, :, ::-1, ::-1]
            # original shape is #count channels rows    columns
            # we want           #count rows     columns channels
            # .swapaxes(1, 2).swapaxes(2, 3)
            weights.append(weight)
            mempointer += filter_size
            biases.append(memmap[mempointer:mempointer + bias_shape])
            mempointer += bias_shape
        #: the network's weights
        self.weights = numpy.array(weights)
        #: the network's biases
        self.biases = numpy.array(biases)

        super(Params, self).__init__()

    @property
    def shape(self):
        """
        The shape of the image.

        Returns
        -------
        shape : tuple
            A tuple of two elements: width and height.
        """
        return (LARGE_INPUT, LARGE_INPUT) if self.large else (SMALL_INPUT,
                                                              SMALL_INPUT)

    @property
    def filter_shapes(self):
        """
        The shape of the filters.

        Returns
        -------
        shape : tuple

        """
        if self.large:
            return LARGE_NETWORK_FILTER_SHAPES
        else:
            return SMALL_NETWORK_FILTER_SHAPES

    @property
    def biases_shapes(self):
        """
        The shape of the biases.

        Returns
        -------
        shape : tuple

        """
        if self.large:
            return LARGE_NETWORK_BIAS_SHAPES
        else:
            return SMALL_NETWORK_BIAS_SHAPES

    def layers(self, large=None):
        """
        Creates the list of layers for MLP model.

        Parameters
        ----------
        large : bool
            The variant - large or small; by default, the value stored in
            the instance is used.

        Returns
        -------
        lylist : list
            A list of layers.
        """
        laylist = []

        if large is None:
            large = self.large

        if large:
            laylist.append(ConvRectifiedLinear(
                output_channels=96,
                kernel_shape=(7, 7),
                kernel_stride=(2, 2),
                pool_shape=(3, 3),
                pool_stride=(3, 3),
                layer_name='h0',
                border_mode='valid',
                pool_type='max',
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ConvRectifiedLinear(
                output_channels=256,
                kernel_shape=(7, 7),
                kernel_stride=(1, 1),
                pool_shape=(2, 2),
                pool_stride=(2, 2),
                layer_name='h1',
                border_mode='valid',
                pool_type='max',
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ZeroPad(padding=1))
            laylist.append(ConvRectifiedLinear(
                output_channels=512,
                kernel_shape=(3, 3),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h2',
                border_mode='full',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ZeroPad(padding=1))
            laylist.append(ConvRectifiedLinear(
                output_channels=512,
                kernel_shape=(3, 3),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h3',
                border_mode='full',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ZeroPad(padding=1))
            laylist.append(ConvRectifiedLinear(
                output_channels=1024,
                kernel_shape=(3, 3),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h4',
                border_mode='full',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ZeroPad(padding=1))
            laylist.append(ConvRectifiedLinear(
                output_channels=1024,
                kernel_shape=(3, 3),
                kernel_stride=(1, 1),
                pool_shape=(3, 3),
                pool_stride=(3, 3),
                layer_name='h5',
                border_mode='full',
                pool_type='max',
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ZeroPad(padding=1))
            laylist.append(ConvRectifiedLinear(
                output_channels=4096,
                kernel_shape=(5, 5),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h6',
                border_mode='valid',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ConvRectifiedLinear(
                output_channels=4096,
                kernel_shape=(1, 1),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h7',
                border_mode='valid',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))
        else:
            laylist.append(ConvRectifiedLinear(
                output_channels=96,
                kernel_shape=(11, 11),
                kernel_stride=(4, 4),
                pool_shape=(2, 2),
                pool_stride=(2, 2),
                layer_name='h0',
                border_mode='valid',
                pool_type='max',
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ConvRectifiedLinear(
                output_channels=256,
                kernel_shape=(5, 5),
                kernel_stride=(1, 1),
                pool_shape=(2, 2),
                pool_stride=(2, 2),
                layer_name='h1',
                border_mode='valid',
                pool_type='max',
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ZeroPad(padding=1))
            laylist.append(ConvRectifiedLinear(
                output_channels=512,
                kernel_shape=(3, 3),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h2',
                border_mode='full',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ZeroPad(padding=1))
            laylist.append(ConvRectifiedLinear(
                output_channels=1024,
                kernel_shape=(3, 3),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h3',
                border_mode='full',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ZeroPad(padding=1))
            laylist.append(ConvRectifiedLinear(
                output_channels=1024,
                kernel_shape=(3, 3),
                kernel_stride=(1, 1),
                pool_shape=(2, 2),
                pool_stride=(2, 2),
                layer_name='h4',
                border_mode='full',
                pool_type='max',
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ConvRectifiedLinear(
                output_channels=3072,
                kernel_shape=(6, 6),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h5',
                border_mode='valid',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))
            laylist.append(ConvRectifiedLinear(
                output_channels=4096,
                kernel_shape=(1, 1),
                kernel_stride=(1, 1),
                pool_shape=None,
                pool_stride=None,
                layer_name='h6',
                border_mode='valid',
                pool_type=None,
                tied_b=True,
                irange=0.0
            ))

        laylist.append(ConvElemwise(
            output_channels=1000,
            kernel_shape=(1, 1),
            kernel_stride=(1, 1),
            nonlinearity=IdentityConvNonlinearity(),
            pool_shape=None,
            pool_stride=None,
            layer_name='h%d' % len(laylist),
            border_mode='full',
            pool_type=None,
            tied_b=True,
            irange=0.0
        ))
        laylist.append(Softmax(
            max_col_norm=1.9365,
            layer_name='y',
            binary_target_dim=1,
            n_classes=1000,
            irange=.005
        ))

        return laylist

    def model(self, large=None, last_layer=None, seed=None):
        """
        Creates the MLP model based on internal attributes.

        Parameters
        ----------
        large : bool, optional
            The variant - large or small; by default, the value stored in
            the instance is used.
        last_layer : optional
            Last layer in the network
        seed : optional
            Seed for random number generator

        Returns
        -------
        model : pylearn2.models.mlp.MLP
            The model
        """
        laylist = self.layers()
        model = MLP(layers=laylist,
                    input_space=Conv2DSpace(
                        shape=self.shape,
                        num_channels=3,
                        axes=['b', 0, 1, 'c']),
                    seed=seed)

        last_layer_std = None
        index = 0
        for lay in laylist[:last_layer_std]:
            if not isinstance(lay, (ZeroPad, Softmax)):
                # we simulate a get_weights method here as
                # the class does not provides one
                # It does provide a get_weights_topo() but that is useless
                # as the shape is changed
                # example:
                #    get_weights => (96, 3, 7, 7)
                #    get_weights_topo => (96, 7, 7, 3)
                crt_w = lay.transformer.get_params()[0].get_value()
                #crt_w = lay.get_weights_topo()
                crt_b = lay.get_biases()
                assert all([crt == new for crt, new in safe_zip(
                    crt_w.shape, self.weights[index].shape)])
                assert all([crt == new for crt, new in safe_zip(
                    crt_b.shape, self.biases[index].shape)])
                lay.set_weights(self.weights[index])
                lay.set_biases(self.biases[index])
                index = index + 1

        return model


class ZeroPad(Layer):
    """
    A layer that adds borders consisting of zeros to its input.

    This is a MLP compatible layer.
    """
    def __init__(self, padding, layer_name=None, *args, **kwargs):
        #: the input space
        self.input_space = None
        #: the output space
        self.output_space = None
        if isinstance(padding, numbers.Number):
            padding = (padding, )
        #: the padding to apply (one value for each axes)
        self.padding = padding
        if layer_name is None:
            layer_name = 'zeropad_%d' % int(round(time.time() * 1000000))
        #: (MLP compatible) layer name
        self.layer_name = layer_name
        super(ZeroPad, self).__init__(*args, **kwargs)
        self.padding_ = padding

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        if isinstance(space, VectorSpace):
            self.output_space = VectorSpace(
                dim=space.dim,
                sparse=space.sparse,
                dtype=space.dtype)
        elif isinstance(space, Conv2DSpace):
            szpad = len(self.padding)
            if szpad == 1:
                self.padding = (self.padding[0], self.padding[0])
            elif szpad != 2 and szpad != 4:
                raise ValueError('ZeroPad layer %s with Conv2DSpace as '
                                 'input space expects a one or two-element '
                                 'padding, nota a %d element'
                                 % (self.layer_name, len(self.padding)))
            if szpad != 4:
                shape = (space.shape[0] + self.padding[0]*2,
                         space.shape[1] + self.padding[1]*2)
                self.padding = [0 if axx == 'c' or axx == 'b'
                                else self.padding[axx]
                                for axx in space.axes]
            else:
                row = space.axes.index(0)
                col = space.axes.index(1)
                shape = (space.shape[row] + self.padding[row]*2,
                         space.shape[col] + self.padding[col]*2)
            self.output_space = Conv2DSpace(
                shape=shape,
                num_channels=space.num_channels,
                axes=space.axes,
                dtype=space.dtype)
        else:
            raise ValueError('ZeroPad layer %s does not support %s '
                             'as input space'
                             % (str(space.__class__), self.layer_name))

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        input_shape = state_below.shape
        if isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()
        elif isinstance(self.input_space, Conv2DSpace):
            output_shape = [self.padding[i]*2+input_shape[i] for i in range(4)]
            output = tensor.zeros(output_shape, dtype=state_below.dtype)

            expression = tensor.set_subtensor(
                output[self.padding[0]:input_shape[0]+self.padding[0],
                       self.padding[1]:input_shape[1]+self.padding[1],
                       self.padding[2]:input_shape[2]+self.padding[2],
                       self.padding[3]:input_shape[3]+self.padding[3]],
                state_below)
        return expression

    @wraps(Model.get_params)
    def get_params(self):
        return []


class ZeroPadYaml(ZeroPad):
    """
    Wrapper for ZeroPad to allow it to be used in yaml.
    """
    def __init__(self, padding, layer_name=None):
        super(ZeroPadYaml, self).__init__(padding, layer_name)


class StandardizePrep(Preprocessor):

    """
    Normalize a dataset.

    The Standardize class in pylearn2 does not allow us to set the
    parameters.

    Parameters
    ----------
    mean : float
        The value is substracted from all elements of the array.
    std : float
        All elements of the array are divided by this value.
    """
    def __init__(self, mean=STANDARDIZE_MEAN, std=STANDARDIZE_STD):
        self.mean = mean
        self.std = std
        assert self.std != 0.0

    @wraps(Preprocessor.apply)
    def apply(self, dataset, can_fit=True):
        X = dataset.get_design_matrix()
        X = (X - self.mean) / self.std
        dataset.set_design_matrix(X)


def standardize(image, mean=STANDARDIZE_MEAN, std=STANDARDIZE_STD):
    """
    Normalize an image or array.

    Parameters
    ----------
    image : str, Image or numpy.ndarray
        The image to process; if a string, it is interpreted to be a path
        towards an image file that is read and converted to numpy array;
        if it is an image is converted to a numpy array. A numpy array
        may also be povided, in which case the input is no longer limited
        to a single image.
    mean : float
        The value is substracted from all elements of the array.
    std : float
        All elements of the array are divided by this value.

    Returns
    -------
    image : numpy.ndarray
        The result of ``(X - mean) / std``.
    """
    if isinstance(image, basestring):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = numpy.array(image.convert('RGB'))
    else:
        assert isinstance(image, numpy.ndarray)
    image = numpy.cast[floatX](image)
    image = (image - mean) / std

    return image

def predict(images, model=None, mapper=get_overfeat_class_label):
    """
    Run a set of images through the network and predict their classes.

    The images provided should match expected size for the model. By
    default the large variant is used, which expects images to be 221
    by 221 pixels.

    The ``images`` parameter is quite flexible, allowing .csv input, single
    and multiple images, PIL images and numpy arrays. Note, however, that
    a single string will not be interpreted as the path to an image file
    but as the path towards a .csv file.

    Examples
    --------

    ::code..

        predict(Image.open('path/to/file.png))

        numpy_img = numpy.zeros(shape=(221, 221, 3), dtype=int)
        predict(numpy_img)

        params = Params(large=False)
        predict('path/to/csv/file.csv', model=params.model())

    Parameters
    ----------
    images : dataset, string, list, tuple, numpy.ndarray, Image.Image
        The image(s) to predict the labels for. This argument is directly
        used with the :class:`Images` constructor if it is not already a
        dataset (it has only been tested with ``DenseDesignMatrix``
        subclasses). Note that - if this is a dataset - the preprocessor
        is NOT applied and you have to apply it yourself.
    model :  pylearn2.models.model.Model, optional
        The model to use; by default a new model is created with
        parameters being initialized from 'large' file.
    mapper : callable, optional
        A function that takes one parameter - an integer - and returns
        the name of the class for that integer.

    Returns
    -------
    probabilities : numpy.ndarray
        The array that was generated by the final layer of the network.
        It contains one entry for each image (first axis).
        Each entry consists of a list of probabilities, one for each class
        (second axis).
    classes : list
        The index of the class with highest probability for each image.
    class_names : list
        The name of the class for each image if a ``mapper`` was provided,
        ``None`` otherwise.
    """

    if model is None:
        params = Params(large=True, weights_file=None)
        model = params.model()

    data_space = model.get_input_space()
    assert isinstance(data_space, Conv2DSpace)
    assert data_space.shape[0] == data_space.shape[1]
    data_source = model.get_input_source()
    data_specs = (data_space, data_source)
    out_space = model.get_output_space()
    class_count = out_space.get_origin_batch(1).shape[1]
    batch = data_space.make_theano_batch('X')
    pfunc = theano.function([batch], model.fprop(batch))

    if isinstance(images, Dataset):
        dataset = images
    else:
        dataset = Images(images, classes=class_count,
                         image_size=data_space.shape[0],
                         preprocessor=StandardizePrep())
    dset_sz = dataset.get_num_examples()
    batch_size = min(dset_sz, 256)
    while batch_size > 0:
        if dset_sz % batch_size == 0:
            break
        batch_size = batch_size - 1
    batch_size = 1
    iter = dataset.iterator(mode='sequential',
                            batch_size=batch_size,
                            data_specs=data_specs)

    probabilities = out_space.get_origin_batch(dset_sz)
    out_idx = 0
    for item in iter:
        result = pfunc(item)
        probabilities[out_idx:out_idx+batch_size] = result
        out_idx = out_idx + batch_size

    # forward propagation results in an array for each example that
    # has a probability for each class
    # we get an array of indices for classes, one for each example
    classes = probabilities.argmax(1)
    assert classes.shape == (dset_sz,)
    if not mapper is None:
        class_names = []
        for cls in classes:
            class_names.append(mapper(cls))
    else:
        class_names = None

    return probabilities, classes, class_names
