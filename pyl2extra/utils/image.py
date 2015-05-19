"""
Utility components for working with images.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan", "Mansour Moufid"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import numpy
from numpy import complex64, real, zeros as _zeros
from numpy.fft import fft2, ifft2, fftshift, ifftshift

#  _zeropad2() and interp2() were copied from fftresize library:
# https://bitbucket.org/eliteraspberries/fftresize/overview
# Following license govern their use:
#    Copyright 2013, 2014, Mansour Moufid <mansourmoufid@gmail.com>
#    
#    Permission to use, copy, modify, and/or distribute this software for any
#    purpose with or without fee is hereby granted, provided that the above
#    copyright notice and this permission notice appear in all copies.
#    
#    THIS SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


def _zeropad2(x, shape):
    '''Pad a two-dimensional NumPy array with zeros along its borders
    to the specified shape.
    '''
    m, n = x.shape
    p, q = shape
    assert p > m
    assert q > n
    tb = (p - m) / 2
    lr = (q - n) / 2
    xpadded = numpy.zeros(shape, dtype=complex64)
    xpadded[tb:tb + m, lr:lr + n] = x
    return xpadded


def interp2(array, factor):
    '''Interpolate a two-dimensional NumPy array by a given factor.
    '''
    reshape = lambda (x, y): [int(factor * x), int(factor * y)]
    diff = lambda (x, y): [x - array.shape[0], y - array.shape[1]]
    nexteven = lambda x: x if (x % 2 == 0) else x + 1
    delta = map(nexteven, diff(reshape(array.shape)))
    newsize = tuple(x[0] + x[1] for x in zip(array.shape, delta))
    fft = fft2(array)
    fft = fftshift(fft)
    fft = _zeropad2(fft, newsize)
    ifft = ifftshift(fft)
    ifft = ifft2(ifft)
    ifft = real(ifft)
    return ifft


def brescale(inp_batch, width, height, axes=('b', 0, 1, 'c')):
    """
    Rescales all images in a batch to requested size.
    
    Parameters
    ----------
    inp_batch : numpy.ndarray
        Input batch; ``len(inp_batch.shape)`` must be 4.
    width : float
        New width.
    height : float
        New height.
    axes : tuple, optional
        Some permutation of b (bathc), 0 (hows, height), 1 (columns, width), 
        c (channels).
    
    Returns
    -------
    out_batch : numpy.ndarray
        Resulted batch with images resized.
    """
    
    