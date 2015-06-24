# -*- coding: utf-8 -*-
"""
Testing helpers related to images.
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
from collections import OrderedDict

from pyl2extra.datasets.images import Images

uniq_files = 0

def create(path, factor=1, shape=None):
    """
    Creates a number of images for testing using various encodings.

    The result is a dictionary with keys being file names and values
    being  tuples of (category, image).
    """
    result = OrderedDict()
    if not os.path.isdir(path):
        os.mkdir(path)

    def mkimg(i, img_base, loc_shp, imgform):
        """Create  a single file"""
        global uniq_files
        img_file = os.path.join(path, img_base % (i+uniq_files))
        wid, hgh = loc_shp if shape is None else shape
        imarray = numpy.random.rand(wid, hgh, 3) * 255
        img = Image.fromarray(imarray.astype('uint8')).convert()
        img.save(img_file)
        uniq_files = uniq_files + 1
        return img_file, img

    for i in range(factor):
        img_file, img = mkimg(i, 'rgba_file_%d.png', (100, 100), 'RGBA')
        result[img_file] = ('rgba', img, 0)
        img_file, img = mkimg(i, 'rgb_file_%d.png', (100, 50), 'RGB')
        result[img_file] = ('rgb', img, 1)
        img_file, img = mkimg(i, 'greyscale_file_%d.png', (50, 100), 'L')
        result[img_file] = ('l', img, 2)
        img_file, img = mkimg(i, 'black_white_file_%d.png', (100, 10), '1')
        result[img_file] = ('bw', img, 3)
        img_file, img = mkimg(i, 'rpalette_file_%d.png', (10, 100), 'P')
        result[img_file] = ('palette', img, 4)
        img_file, img = mkimg(i, 'cmyk_file_%d.jpg', (255, 254), 'CMYK')
        result[img_file] = ('cmyk', img, 5)
        img_file, img = mkimg(i, 'integer_file_%d.png', (10, 11), 'I')
        result[img_file] = ('integer', img, 6)
        img_file, img = mkimg(i, 'float_file_%d.tif', (999, 999), 'F')
        result[img_file] = ('float', img, 7)
    return result

def dataset(image_size, path, factor=1):
    """
    Creates a dataset of images.
    """

    images = create(path, factor)

    # create a dictionary mapping images to classes
    inpimg = OrderedDict()
    for img in images:
        inpimg[images[img][1]] = images[img][2]

    return Images(source=inpimg,
                  image_size=image_size,
                  classes=8,
                  rng=None,
                  preprocessor=None,
                  fit_preprocessor=False)
