"""
Helper functions related to numpy.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from copy import copy
import numpy


def _slice(array, outlen, address=None):
    """
    Implementation for the slice_Xd functions.
    """
    def _loc_address(i):
        """Get the address by using previous one and current index"""
        if address is None:
            loc_addr = [i]
        else:
            loc_addr = copy(address)
            loc_addr.append(i)
        return loc_addr

    if len(array.shape) > outlen+1:
        for i in range(array.shape[0]):
            loc_addr = _loc_address(i)
            for addr, arr in _slice(array[i], outlen, loc_addr):
                yield addr, arr
    else:
        for i in range(array.shape[0]):
            loc_addr = _loc_address(i)
            yield loc_addr, array[i]

def slice_1d(array):
    """
    Yields all 1D components of the array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to slice.

    Returns
    -------
    slice : generator
        A generator that yelds a pair of values at a time. First value is a
        list indicating the address of this componect and second value is
        the component itself.
    """
    assert isinstance(array, numpy.ndarray)
    assert len(array.shape) >= 1
    return _slice(array, 1)

def slice_2d(array):
    """
    Yields all 2D components of the array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to slice.

    Returns
    -------
    slice : generator
        A generator that yelds a pair of values at a time. First value is a
        list indicating the address of this componect and second value is
        the component itself.
    """
    assert isinstance(array, numpy.ndarray)
    assert len(array.shape) >= 1
    return _slice(array, 2)

def slice_3d(array):
    """
    Yields all 3D components of the array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to slice.

    Returns
    -------
    slice : generator
        A generator that yelds a pair of values at a time. First value is a
        list indicating the address of this componect and second value is
        the component itself.
    """
    assert isinstance(array, numpy.ndarray)
    assert len(array.shape) >= 1
    return _slice(array, 3)
