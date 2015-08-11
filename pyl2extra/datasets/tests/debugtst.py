#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
os.environ['THEANO_FLAGS'] ='optimizer=fast_compile,exception_verbosity=high,device=cpu,floatX=float64,allow_gc=True'
import tempfile
import theano
import numpy
import shutil

from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.train_extensions import window_flip
from pylearn2.utils import serial
from pylearn2.train import Train
from pylearn2.space import Conv2DSpace

from pyl2extra.datasets.images import Images
from pyl2extra.testing import images

print("Theano device: %s" % (theano.config.device))
print("Theano mode: %s" % (theano.config.mode))
print("Theano optimizer: %s" % (theano.config.optimizer))


def create_csvs(tmp_dir, testset):
    """
    """
    csv_file_fc = os.path.join(tmp_dir, 'files_and_categs.csv')
    with open(csv_file_fc, 'wt') as fhand:
        csvw = csv.writer(fhand, delimiter=',', quotechar='"')
        for fpath in testset:
            csvw.writerow([testset[fpath][2], fpath])
    csv_file_f = os.path.join(tmp_dir, 'files.csv')
    with open(csv_file_f, 'wt') as fhand:
        csvw = csv.writer(fhand, delimiter=',', quotechar='"')
        for fpath in testset:
            csvw.writerow(['', fpath])
    return csv_file_fc, csv_file_f


mydataset = True
# change the scale of the network
network_scale = 1
output_scale = 5*5
IMAGE_SHAPE = 64


if mydataset:
    tmp_dir = tempfile.mkdtemp()
    images = images.create(tmp_dir, factor=10)
    csv_file_fc, csv_file_f = create_csvs(tmp_dir, images)
    trn = Images(source=csv_file_fc, image_size=IMAGE_SHAPE, regression=False)
    CLASS_COUNT = 8
    print tmp_dir, csv_file_fc, csv_file_f
else:
    from pylearn2.datasets.mnist import MNIST
    trn = MNIST(which_set= 'train',
        axes=['c', 0, 1, 'b'],
        start=0,
        stop=50000
    )
    trn.axes = ['c', 0, 1, 'b']
    IMAGE_SHAPE = 28
    CLASS_COUNT = 10



# input shape must be like this (!)
in_space = Conv2DSpace(shape=(IMAGE_SHAPE, IMAGE_SHAPE),
                       num_channels=1,
                       axes=trn.axes)


# choose the function based on architecture
if theano.config.device == 'cpu':
    layers_func = maxout.MaxoutLocalC01B
    have_gpu = False
else:
    layers_func = maxout.MaxoutConvC01B
    have_gpu = True

irange = 0.001

# first layer
layer_1 = layers_func(layer_name='layer_1',
                      pad=0,
                      tied_b=1,
                      W_lr_scale=.05,
                      b_lr_scale=.05,
                      num_channels=network_scale*16,
                      num_pieces=2,
                      kernel_shape=(8, 8),
                      pool_shape=((4, 4) if have_gpu else None),
                      pool_stride=((2, 2) if have_gpu else None),
                      irange=irange,
                      max_kernel_norm=.9,
                      partial_sum=1)

# second layer
#layer_2 = layers_func(layer_name='layer_2',
#                      pad=0,
#                      tied_b=1,
#                      W_lr_scale=.05,
#                      b_lr_scale=.05,
#                      num_channels=network_scale*2*16,
#                      num_pieces=2,
#                      kernel_shape=(8, 8),
#                      pool_shape=((4, 4) if have_gpu else None),
#                      pool_stride=((2, 2) if have_gpu else None),
#                      irange=irange,
#                      max_kernel_norm=1.9365,
#                      partial_sum=1)
# third layer
#layer_3 = layers_func(layer_name='layer_3',
#                      pad=0,
#                      tied_b=1,
#                      W_lr_scale=.05,
#                      b_lr_scale=.05,
#                      num_channels=network_scale*2*16,
#                      num_pieces=2,
#                      kernel_shape=(5, 5),
#                      pool_shape=((2, 2) if have_gpu else None),
#                      pool_stride=((2, 2) if have_gpu else None),
#                      irange=irange,
#                      max_kernel_norm=1.9365)
# fourth layer
#layer_4 = maxout.Maxout(layer_name='layer_4',
#                        irange=irange,
#                        num_units=output_scale,
#                        num_pieces=5,
#                        max_col_norm=1.9)

# fifth (output) layer
output = mlp.Softmax(layer_name='y',
                     n_classes=CLASS_COUNT,
                     irange=irange,
                     max_col_norm=1.9365)

mdl = mlp.MLP([layer_1, output],
              input_space=in_space)

trainer = sgd.SGD(learning_rate=.05,
                  batch_size=64,
                  learning_rule=learning_rule.Momentum(0.10),
                  cost=Dropout(),
                  monitoring_dataset={'train': trn})

#tv = trn.get_topological_view()
#trn.set_topological_view(tv, axes=trn.view_converter.axes)

win = window_flip.WindowAndFlip(pad_randomized=8,
                                window_shape=(IMAGE_SHAPE, IMAGE_SHAPE),
                                randomize=[trn],
                                center=[trn])
experiment = Train(dataset=trn,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[])

experiment.main_loop()











if mydataset:
    shutil.rmtree(tmp_dir)

