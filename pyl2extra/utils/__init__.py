"""
Assorted utility methods that found no other place.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

def slice_count(slc):
    """
    Computes the number of elements in a slice object.
    """
    cnt = slc.stop - slc.start
    if not slc.step is None:
        if cnt % slc.step > 0:
            cnt = cnt / slc.step + 1
        else:
            cnt = cnt / slc.step
    return cnt
