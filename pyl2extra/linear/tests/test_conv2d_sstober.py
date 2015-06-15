import os
import numpy

os.environ['THEANO_FLAGS'] = 'mode=DebugMode,exception_verbosity=high,optimizer=None'

from pylearn2.linear.conv2d import Conv2D
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng
from pylearn2.space import Conv2DSpace
import theano

from theano.printing import pp, pprint, pydotprint

default_seed = [2012, 11, 6, 9]
rng = make_np_rng(None, default_seed, which_method='uniform')
 
irange = 0.05
input_channels = 1
output_channels = 1
kernel_shape = [2,2]
kernel_stride = [2,2] # NOTE: this works for [1,1]
 
input_space = Conv2DSpace(shape = (8, 8), num_channels=input_channels, axes = ('b', 'c', 0, 1))
 
# create random filter matrix in bc01 layout
W = sharedX(rng.uniform(
                -irange, irange,
                (output_channels, input_channels, kernel_shape[0], kernel_shape[1])
            ), name="W")

X = numpy.array([1, 2, 3, 4, 5, 6, 7, 8]*8, dtype="float32").reshape((1, 1, 8, 8))
conv = Conv2D(
        filters=W,
        batch_size=1,
        input_space=input_space,
        output_axes=('b', 'c', 0, 1),
        subsample=kernel_stride
    )
#conv._border_mode = 'full'

x = input_space.get_theano_batch()
x.name = 'x'
#y = input_space.get_theano_batch()
#y.name = 'y'
outvar = conv.lmul_T(conv.lmul(x))
pydotprint(outvar, outfile='/media/tnick/Big_mamma1/prog/python/pylearn2/pyl2extra/pyl2extra/linear/tests/outvar.png')

#outvar.eval({x: X})
f = theano.function(inputs=[x], 
                    outputs=outvar,
                    mode=None,
                    updates=None,
                    givens=None,
                    no_default_updates=False, 
                    accept_inplace=False,
                    name='test_function',
                    rebuild_strict=True, 
                    allow_input_downcast=None, 
                    profile=None,
                    on_unused_input=None)
pydotprint(f, outfile='/media/tnick/Big_mamma1/prog/python/pylearn2/pyl2extra/pyl2extra/linear/tests/function.png')
print f(X)

"""
The last line will throw a MissingInputError (with theano exception_verbosity='high'):

MissingInputError: A variable that is an input to the graph was neither provided as an input to 
the function nor given a value. A chain of variables leading from this input to an output is 
[dummy_v, shuffle_for_conv3D(dummy_v), Subtensor{int64, int64, int64, int64, ::}.0, 
Elemwise{second,no_inplace}.0, <theano.tensor.nnet.ConvTransp3D.ConvTransp3D object at 0x10eb48710>.0, 
Conv3D_dCdV(dCdH=anon_dCdH,V=shuffle_for_conv3D(dummy_v)), Elemwise{identity}.0, DimShuffle{0,4,1,2}.0]. 
This chain may not be unique
Backtrace when the variable is created:
  File "/Users/sstober/git/pylearn2/pylearn2/linear/conv2d.py", line 159, in lmul_T
    dummy_v = T.tensor4()
"""
