"""
Tests for cos_dataset module.
"""
from pyl2extra.datasets.cos_dataset import CosDataset, _banned_mods
from pylearn2.utils.iteration import _iteration_schemes
from pylearn2.utils.rng import make_np_rng
from pylearn2.space import VectorSpace
from nose.tools import assert_raises

from itertools import chain

def test_constructor():
    
    cdset = CosDataset()
    print cdset
    
def test_iterator():
    
    std_dev = 0.005
    cdset = CosDataset(std=std_dev)
    
    indiv_ex = range(100)
    cklim = 1 + std_dev*6
    def is_in_valid_range(y):
        result = True
        for y1 in y:
            for y2 in y1:
                result = result and y2 < cklim and y2 > -cklim
                #print result, -cklim, y2, cklim
                
        #result = np.all([np.all(yval < cklim and yval > -cklim) for yval in y])
        #print zip(y, [np.all(yval < cklim and yval > -cklim) for yval in y])        
        return result
    
    itr = cdset.iterator()
    for i in indiv_ex:
        try:
            y = itr.next()
        except StopIteration:
            itr = cdset.iterator()
            y = itr.next()
        #print y, list(y <= 1.05 and y >= -1.05)
        assert is_in_valid_range(y)

    for imode in chain(_iteration_schemes, ['random']):
        #print imode, imode in _banned_mods, _banned_mods
        if imode in _banned_mods:
            assert_raises(ValueError, cdset.iterator, mode=imode)
        else:
            itr = cdset.iterator(mode=imode)
            for i in indiv_ex:
                try:
                    y = itr.next()
                except StopIteration:
                    itr = cdset.iterator(mode=imode)
                    y = itr.next()
                assert is_in_valid_range(y)
    
    for batchsz in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
        print batchsz
        itr = cdset.iterator(batch_size=batchsz)
        for i in indiv_ex:
            try:
                y = itr.next()
            except StopIteration:
                itr = cdset.iterator(batch_size=batchsz)
                y = itr.next()
                
            assert is_in_valid_range(y)
        
    rng = make_np_rng((1, 2, 3), which_method=['uniform', 'randn'])
    itr = cdset.iterator(mode='random', rng=rng)
    for i in indiv_ex:
        try:
            y = itr.next()
        except StopIteration:
            itr = cdset.iterator(mode='random', rng=rng)
            y = itr.next()
        assert is_in_valid_range(y)
        
    data_specs = (VectorSpace(2), 'features')
    itr = cdset.iterator(data_specs=data_specs)
    for i in indiv_ex:
        try:
            y = itr.next()
        except StopIteration:
            itr = cdset.iterator(data_specs=data_specs)
            y = itr.next()
        assert is_in_valid_range(y)
        
    itr = cdset.iterator(return_tuple=True)
    for i in indiv_ex:
        try:
            y = itr.next()
        except StopIteration:
            itr = cdset.iterator(return_tuple=True)
            y = itr.next()
        assert isinstance(y, tuple)
    
if __name__ == "__main__":
    test_iterator()
