# -*- coding: utf-8 -*-
"""
Layers and models based on OverFeat_ weights.

OverFeat_ is a Convolutional Network-based image classifier
and feature extractor. OverFeat is Copyright NYU 2013. Authors are
Michael Mathieu, Pierre Sermanet, and Yann LeCun. The weights archive
is provided separatelly (here <http://cilvr.cs.nyu.edu/lib/exe/fetch.php?media=overfeat:overfeat-weights.tgz>)_

..code:

    wget %s -O weights.tgz
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

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ['Michael Mathieu', 'Pierre Sermanet', 'Yann LeCun',
               'Kyle Kastner', 'Michael Eickenberg', 'Nicu Tofan']
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import Image
import os
import numpy
import theano
from pylearn2.models.mlp import (MLP, IdentityConvNonlinearity, ConvElemwise,
                                 ConvRectifiedLinear, SigmoidConvNonlinearity)
from pylearn2.space import Conv2DSpace
from theano import tensor

floatX = theano.config.floatX

# characteristics of the small network
SMALL_NETWORK_WEIGHT_FILE = 'net_weight_0'
SMALL_NETWORK_FILTER_SHAPES = numpy.array([(96, 3, 11, 11),
                                           (256, 96, 5, 5),
                                           (512, 256, 3, 3),
                                           (1024, 512, 3, 3),
                                           (1024, 1024, 3, 3),
                                           (3072, 1024, 6, 6),
                                           (4096, 3072, 1, 1),
                                           (1000, 4096, 1, 1)])
SMALL_NETWORK_BIAS_SHAPES = SMALL_NETWORK_FILTER_SHAPES[:, 0]
SMALL_NETWORK = (SMALL_NETWORK_WEIGHT_FILE,
                 SMALL_NETWORK_FILTER_SHAPES,
                 SMALL_NETWORK_BIAS_SHAPES)
SMALL_INPUT = 231

# characteristics of the large network
LARGE_NETWORK_WEIGHT_FILE = 'net_weight_1'
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
LARGE_NETWORK = (LARGE_NETWORK_WEIGHT_FILE,
                 LARGE_NETWORK_FILTER_SHAPES,
                 LARGE_NETWORK_BIAS_SHAPES)
LARGE_INPUT = 221

FILE_PATH_KEY = 'PYL2X_OVERFEAT_PATH'

class Params(object):
    """
    Container for OverFeat weights and biases.

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
        #: The shapes for filters
        self.filter_shapes = LARGE_NETWORK_FILTER_SHAPES \
            if large else SMALL_NETWORK_FILTER_SHAPES
        #: The shapes for biases
        self.biases_shapes = LARGE_NETWORK_BIAS_SHAPES \
            if large else SMALL_NETWORK_BIAS_SHAPES

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
            weights.append(wflat.reshape(weight_shape)[:, :, ::-1, ::-1])
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
        if large is None:
            large = self.large
        lylist = []
        if large:
            lylist.append(self.conv_layer(0, 2, 3, 'valid'))
            lylist.append(self.conv_layer(1, 1, 2, 'valid',
                                          zpad=True))
            lylist.append(self.conv_layer(2, 1, None, 'full',
                                          zpad=True))
            lylist.append(self.conv_layer(3, 1, None, 'full',
                                          zpad=True))
            lylist.append(self.conv_layer(4, 1, None, 'full',
                                          zpad=True))
            lylist.append(self.conv_layer(5, 1, 3, 'full',
                                          zpad=True))
            lylist.append(self.conv_layer(6, 1, None, 'valid'))
            lylist.append(self.conv_layer(7, 1, None, 'valid'))
        else:
            lylist.append(self.conv_layer(0, 4, 2, 'valid'))
            lylist.append(self.conv_layer(1, 1, 2, 'valid',
                                          zpad=True))
            lylist.append(self.conv_layer(2, 1, None, 'full',
                                          zpad=True))
            lylist.append(self.conv_layer(3, 1, None, 'full',
                                          zpad=True))
            lylist.append(self.conv_layer(4, 1, 2, 'full'))
            lylist.append(self.conv_layer(5, 1, None, 'valid'))
            lylist.append(self.conv_layer(6, 1, None, 'valid'))
        return lylist
        
    def model(self, seed=None, last_layer=None):
        """
        Creates the MLP model based on internal attributes.

        Returns
        -------
        model : pylearn2.models.mlp.MLP
            The model
        """
        lylist = self.layers()
        if last_layer is None:
            last_layer_std = None
            output_channels = self.filter_shapes[-1, 0]
            last_layer = ConvElemwise(output_channels=output_channels,
                                      kernel_shape=(1, 1),
                                      kernel_stride=(1, 1),
                                      layer_name="y",
                                      #nonlinearity=IdentityConvNonlinearity(),
                                      nonlinearity=SigmoidConvNonlinearity(),
                                      border_mode='valid',
                                      pool_type=None,
                                      pool_shape=None,
                                      pool_stride=None,
                                      irange=0.0,
                                      tied_b=True)
        else:
            last_layer_std = -1
        lylist.append(last_layer)
        
        model = MLP(layers=lylist,
                    batch_size=None,
                    input_space=Conv2DSpace(shape=self.shape,
                                            num_channels=3,
                                            axes=('b', 0, 1, 'c'),
                                            dtype=floatX),
                    input_source='features',
                    target_source='targets',
                    seed=seed)
        
        for index, lay in enumerate(lylist[:last_layer_std]):
            lay.set_weights(self.weights[index])
            lay.set_biases(self.biases[index])
        
        return model

    def conv_layer(self, index, kernel, pool, border, name=None, zpad=False):
        """
        """
        if pool is None:
            pool_shape = None
        else:
            pool_shape = (pool, pool)
            pool = 'max'
        if name is None:
            name = 'h%d' % index
        output_channels=self.filter_shapes[index, 0]
        out_adjuster = zero_pad if zpad else None
        lay = ConvRectifiedLinear(output_channels=output_channels,
                                  kernel_shape=(kernel, kernel),
                                  kernel_stride=(1, 1),
                                  pool_shape=pool_shape,
                                  pool_stride=pool_shape,
                                  layer_name=name,
                                  border_mode=border,
                                  pool_type=pool,
                                  tied_b=True,
                                  irange=0.0,
                                  output_normalization=out_adjuster)
        return lay


def zero_pad(expr, pad=1):
    """
    Add a border of zeros that is symetrical in all four sides.
    """
    p = (pad, pad, pad, pad)
    input_shape = expr.shape
    output_shape = (input_shape[0], input_shape[1],
                    input_shape[2] + p[0] + p[2],
                    input_shape[3] + p[1] + p[3])
    output = tensor.zeros(output_shape, dtype=expr.dtype)
    return tensor.set_subtensor(
        output[:, :, p[0]:output_shape[2] - p[2],
                     p[1]:output_shape[3] - p[3]],
        expr)

def standardize(image, mean=118.380948, std=61.896913):
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



#def model():
