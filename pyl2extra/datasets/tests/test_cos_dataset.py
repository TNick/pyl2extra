"""
Tests for cos_dataset module.
"""
from pyl2extra.datasets.cos_dataset import CosDataset, _banned_mods
from pylearn2.utils.iteration import _iteration_schemes
from pylearn2.utils.rng import make_np_rng
from pylearn2.space import VectorSpace
import numpy as np
from nose.tools import assert_raises

from itertools import chain

def test_constructor():
    
    cdset = CosDataset()
    print cdset
    
def test_iterator():
    
    cdset = CosDataset()
    
    itr = cdset.iterator()
    indiv_ex = range(100)
    std_dev = 0.05
    cklim = 1 + std_dev*6
    def is_in_valid_range(y):
        return np.all(y < cklim and y > -cklim)
    
    for i in indiv_ex:
        y = itr.next()
        #print y, list(y <= 1.05 and y >= -1.05)
        assert is_in_valid_range(y)

    for imode in chain(_iteration_schemes, ['random']):
        print imode, imode in _banned_mods, _banned_mods
        if imode in _banned_mods:
            assert_raises(ValueError, cdset.iterator, mode=imode)
        else:
            itr = cdset.iterator(mode=imode)
            for i in indiv_ex:
                y = itr.next()
                assert is_in_valid_range(y)
    
    for batchsz in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
        print batchsz
        itr = cdset.iterator(batch_size=batchsz)
        for i in indiv_ex:
            y = itr.next()
            assert is_in_valid_range(y)
        
    rng = make_np_rng((1, 2, 3), which_method=['uniform', 'randn'])
    itr = cdset.iterator(mode='random', rng=rng)
    for i in indiv_ex:
        y = itr.next()
        assert is_in_valid_range(y)
        
    data_specs = (VectorSpace(2), 'source')
    itr = cdset.iterator(data_specs=data_specs)
    for i in indiv_ex:
        y = itr.next()
        assert is_in_valid_range(y)
        
    itr = cdset.iterator(return_tuple=True)
    for i in indiv_ex:
        y = itr.next()
        assert isinstance(y, tuple)
        #print y
        
    #assert False
    