"""
Provides a easy solution for using images in Pylearn2.

Input files may be provided in various ways:

- as a list of dictionaries, where the keys are paths to image files
  and the values are the class for that image;
- as a csv file, where first column is the class and second column is
  the path toward the image file by default;
- in general using an iterator instance that spits file - value pairs.

The images will all be resized to same square dimensions (customizable)
and those in RGBA format will be converted to RGB. The empty spaces generated 
in this fashion will be filled with a color of your choosing or using a
background image.

Various affine transformations can be applied (rotation, scale, flip),
patches can be extracted and the contrast can be normalized.

Preparing the batches can be done in a number of ways:

- online, in the same thread as the requester;
- using a customisable number of threads
- using a customisable number of processes

Online option can be useful if the user wants a new, 
pylearn2.datasets.DenseDesignMatrix created from
a set of files; paralel options are suitable when the proprocessing is done
while previous batch is used for training. The dataset may be forced to use
the CPU for this opperation independent of the environment flags for
Theano.

When preparing batches the class can employ various strategies:
- apply all possible transformations for an image then step to next image;
- apply a random set of transformations to each image in the batch

"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from pyl2extra.datasets.img_dataset.dataset import ImgDataset
