"""
Tests for adjusters.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import functools
import Image, ImageDraw
import numpy
import os
import cPickle
import shutil
import tempfile
import theano
import unittest

from pyl2extra.datasets.img_dataset.adjusters import (FlipAdj, RotationAdj,
                                                      ScalePatchAdj, GcaAdj,
                                                      BackgroundAdj,
                                                      MakeSquareAdj,
                                                      adj_from_string)

from pyl2extra.datasets.img_dataset.data_providers import RandomProvider

def create_rgba_image(width=256, height=256, cornerf=(0, 255, 255, 255)):
    """
    Create a simple image.

    Top right corner has a triangle.
    Top left corner has some text.
    """
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))

    # get a font and a drawing context
    draw = ImageDraw.Draw(image)

    # draw text, half opacity and full opacity
    draw.text((10, 10), "Hello", fill=(0, 255, 255, 128))
    draw.text((10, 60), "World", fill=(0, 255, 255, 255))

    draw.polygon([(width/2, 0), (width-1, height/2),
                  (width-1, 0), (width/2, 0)],
                 fill=cornerf)

    return image

def create_rgb_image(width=256, height=256, cornerf=(64, 0, 128)):
    """
    Create a simple image.

    Bottom right corner has a triangle.
    """
    image = Image.new('RGB', (width, height), (128, 0, 64))
    draw = ImageDraw.Draw(image)
    draw.polygon([(width/2, height-1), (width-1, height-1),
                  (width-1, height/2), (width/2, height-1)],
                 fill=cornerf)
    return image

def create_batch(batch_sz, width=128, height=128):
    """
    Create a test batch
    """
    im_rgba = create_rgba_image(width*2, height*2)
    im_rgba.thumbnail((width, height), Image.ANTIALIAS, )
    im_rgba = numpy.array(im_rgba)
    im_rgba = numpy.cast[theano.config.floatX](im_rgba)

    batch = numpy.empty(shape=(batch_sz, height, width, 4))
    for i in range(batch_sz):
        batch[i, :, :, :] = im_rgba
    return batch

def create_mbatch(batch_sz, width=128, height=128):
    """
    Create a test batch
    """
    def doimg(cornerf):
        """One image"""

        im_rgba = create_rgba_image(width*2, height*2, cornerf=cornerf)
        im_rgba.thumbnail((width, height), Image.ANTIALIAS, )
        im_rgba = numpy.array(im_rgba)
        im_rgba = numpy.cast[theano.config.floatX](im_rgba)
        return im_rgba

    img_array = [doimg(cornerf=(255, 0, 0)),
                 doimg(cornerf=(0, 255, 0)),
                 doimg(cornerf=(0, 0, 255)),
                 doimg(cornerf=(255, 255, 0)),
                 doimg(cornerf=(0, 255, 255)),
                 doimg(cornerf=(255, 0, 255)),
                 doimg(cornerf=(64, 0, 128))]

    batch = numpy.empty(shape=(batch_sz, height, width, 4))
    for i in range(batch_sz):
        batch[i, :, :, :] = img_array[i % len(img_array)]
    return batch

class BaseAdjusters(object):
    """
    Mixin to extract common functionality
    """
    def __init__(self):
        #: the class instance to test
        self.testee = None
        super(BaseAdjusters, self).init()

    def prepare(self):
        """
        Constructor helper.
        """
        self.tmp_dir = tempfile.mkdtemp()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.testee

class TestBackgroundAdj(unittest.TestCase, BaseAdjusters):
    """
    Tests for BackgroundAdj.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.prepare()
        im_bkg = create_rgb_image()
        self.testee = BackgroundAdj(backgrounds=[im_bkg, (250, 0, 250)])
        self.testee.setup(None, 'seq_one')

    def test_pickle(self):
        """
        Make sure we can pickle-unpickle
        """
        im_bkg = create_rgb_image()
        pickle_png = os.path.join(self.tmp_dir, "pickle.png")
        im_bkg.save(pickle_png)
        local_testee = BackgroundAdj(backgrounds=[(250, 0, 250)],
                                     image_files=pickle_png)

        dmp = cPickle.dumps(local_testee)
        reloaded = cPickle.loads(dmp)
        batch_sz = 5
        batch = create_batch(batch_sz)
        batch = reloaded.process(batch)
        self.assertEqual(batch.shape[0], batch_sz)


    def test_transf_count(self):
        """
        Check the transf_count() for BackgroundAdj
        """
        instloc = BackgroundAdj()
        self.assertEqual(instloc.transf_count(), 1)
        self.assertEqual(len(instloc.backgrounds), 1)

    def test_normalize_back(self):
        """
        Check the _normalize_back() for BackgroundAdj
        """
        test_list = BackgroundAdj._normalize_back([], None)
        self.assertEqual(len(test_list), 1)
        self.assertEqual(test_list[0], (0, 0, 0))

        test_list = BackgroundAdj._normalize_back([], (1, 1, 1))
        self.assertEqual(len(test_list), 1)
        self.assertEqual(test_list[0], (1, 1, 1))

        test_list = BackgroundAdj._normalize_back([], 'white')
        self.assertEqual(len(test_list), 1)
        self.assertEqual(test_list[0], (255, 255, 255))

        test_list = BackgroundAdj._normalize_back([], 'black')
        self.assertEqual(len(test_list), 1)
        self.assertEqual(test_list[0], (0, 0, 0))

        test_list = BackgroundAdj._normalize_back([], '#FFFFFF')
        self.assertEqual(len(test_list), 1)
        self.assertEqual(test_list[0], (255, 255, 255))

        inp = ['white', 'black', None, (123, 124, 125), "#FFFFFF"]
        test_list = BackgroundAdj._normalize_back([], inp)
        self.assertEqual(len(test_list), len(inp))
        self.assertEqual(test_list[0], (255, 255, 255))
        self.assertEqual(test_list[1], (0, 0, 0))
        self.assertEqual(test_list[2], (0, 0, 0))
        self.assertEqual(test_list[3], (123, 124, 125))
        self.assertEqual(test_list[4], (255, 255, 255))

    def scan_result_batch(self, batch):
        """
        Common method for testing a bunch of images.
        """
        for i in range(batch.shape[0]):
            im_rgb = batch[i, :, :, :]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if show_im: img = Image.fromarray(im_rgb)
            if show_im: img.show()

            # top right corner should have color from the image
            self.assertEqual(0, im_rgb[2, 120, 0])
            self.assertLessEqual(254, im_rgb[2, 120, 1])
            self.assertLessEqual(254, im_rgb[2, 120, 2])
            if i % 2 == 0:
                # top left corner should have same image as the background
                self.assertLessEqual(127, im_rgb[2, 2, 0])
                self.assertEqual(0, im_rgb[2, 2, 1])
                self.assertLessEqual(63, im_rgb[2, 2, 2])
                # bottom right corner should have color from the background
                self.assertLessEqual(63, im_rgb[120, 120, 0])
                self.assertEqual(0, im_rgb[120, 120, 1])
                self.assertLessEqual(128, im_rgb[120, 120, 2])
            else:
                # top left corner should have same image as the background
                self.assertLessEqual(250, im_rgb[2, 2, 0])
                self.assertEqual(0, im_rgb[2, 2, 1])
                self.assertLessEqual(250, im_rgb[2, 2, 2])
                # bottom right corner should have color from the background
                self.assertLessEqual(250, im_rgb[120, 120, 0])
                self.assertEqual(0, im_rgb[120, 120, 1])
                self.assertLessEqual(250, im_rgb[120, 120, 2])

    def test_perform_image(self):
        """
        Applies an image as a background.
        """
        batch_sz = 5
        batch = create_batch(batch_sz)
        batch = self.testee.process(batch)
        self.scan_result_batch(batch)

    def test_accumulate_image(self):
        """
        Applies an image as a background.
        """
        batch_sz = 5
        batch = create_batch(batch_sz)
        batch = self.testee.accumulate(batch)
        for i in range(batch_sz):
            per_image = batch[i::batch_sz]
            self.assertEqual(per_image.shape,
                             (self.testee.transf_count(), 128, 128, 3))
            self.scan_result_batch(per_image)

    def test_perform_color(self):
        """
        Applies a color as a background.
        """
        batch_sz = 5
        batch = create_batch(batch_sz)
        self.testee.prmstore.next()
        batch = self.testee.process(batch)

        for i in range(batch_sz):
            im_rgb = batch[i, :, :, :]
            im_rgb = numpy.cast['uint8'](im_rgb)

            if show_im: img = Image.fromarray(im_rgb)
            if show_im: img.show()

            # top left corner should have same image as the background
            self.assertLessEqual(127, im_rgb[2, 2, 0])
            self.assertEqual(0, im_rgb[2, 2, 1])
            self.assertLessEqual(63, im_rgb[2, 2, 2])
            # top right corner should have color from the image
            self.assertEqual(0, im_rgb[2, 120, 0])
            self.assertLessEqual(254, im_rgb[2, 120, 1])
            self.assertLessEqual(254, im_rgb[2, 120, 2])


class MockDataset(object):
    """
    Simply mocks the dataset.
    """
    def __init__(self):
        self.shape = None


class TestMakeSquareAdj(unittest.TestCase):
    """
    Tests for MakeSquareAdj.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = MakeSquareAdj()
        dataset = MockDataset()
        self.testee.setup(dataset, 'seq_one')
        self.tmp_dir = tempfile.mkdtemp()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee
        shutil.rmtree(self.tmp_dir)

    def test_transf_count(self):
        """
        Check the transf_count() for MakeSquareAdj
        """
        instloc = MakeSquareAdj()
        self.assertEqual(instloc.transf_count(), 1)

    def test_perform(self):
        """
        Makes batches square.
        """
        batch_sz = 5
        batch = create_batch(batch_sz, 256, 128)
        batch = self.testee.process(batch)

        for i in range(batch_sz):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)

            img = Image.fromarray(im_rgb)
            if show_im: img.show()

    def test_create_ddm_square(self):
        """
        Makes dense design dataset.
        """
        datap = RandomProvider(count=50)
        cache_loc = os.path.join(self.tmp_dir, 'ddm.pkl')
        cache_npy = cache_loc + '.npy'
        ddm = self.testee.create_ddm(datap, cache_loc=cache_loc)

        self.assertTrue(os.path.isfile(cache_loc))
        self.assertTrue(os.path.isfile(cache_npy))
        self.assertEqual(ddm.get_num_examples(), 50)

    def test_create_ddm_non_square(self):
        """
        Makes dense design dataset.
        """
        datap = RandomProvider(count=50, size=(128, 256))
        cache_loc = os.path.join(self.tmp_dir, 'ddm.pkl')
        cache_npy = cache_loc + '.npy'
        ddm = self.testee.create_ddm(datap, cache_loc=cache_loc)

        self.assertTrue(os.path.isfile(cache_loc))
        self.assertTrue(os.path.isfile(cache_npy))
        self.assertEqual(ddm.get_num_examples(), 50)

class TestFlipAdj(unittest.TestCase):
    """
    Tests for FlipAdj.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = FlipAdj()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee

    def test_transf_count(self):
        """
        Check the transf_count() for FlipAdj
        """
        instloc = FlipAdj()
        self.assertEqual(instloc.transf_count(), 2)

        instloc = FlipAdj(horizontal=True, vertical=False)
        self.assertEqual(instloc.transf_count(), 2)

        instloc = FlipAdj(horizontal=False, vertical=True)
        self.assertEqual(instloc.transf_count(), 2)

        instloc = FlipAdj(horizontal=True, vertical=True)
        self.assertEqual(instloc.transf_count(), 4)

    def test_horizontal(self):
        """
        Flip batches only horizontally.
        """
        self.testee.horizontal = True
        self.testee.vertical = False
        self.testee.setup(None, 'seq_one')

        batch_sz = 5
        batchin = create_mbatch(batch_sz, 128, 128)

        im_rgb = batchin[0, :, :, 0:3]
        im_rgb = numpy.cast['uint8'](im_rgb)
        img = Image.fromarray(im_rgb)
        if show_im: img.show()

        batch = self.testee.process(batchin)
        self.assertEqual(batch.shape, (batch_sz, 128, 128, 4))

        for i in range(batch_sz):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i == 0:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(
                    title="TestFlipAdj.test_horizontal %d" % i)

        batch = self.testee.accumulate(batchin)
        self.assertEqual(batch.shape, (batch_sz*2, 128, 128, 4))
        self.assertTrue(numpy.any(batch[0] != batch[batch_sz/2+1]))
        for i in range(batch.shape[0]):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i >= 0:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(title="TestFlipAdj.test_vertical %d" % i)

    def test_vertical(self):
        """
        Flip batches only vertically.
        """
        self.testee.horizontal = False
        self.testee.vertical = True
        self.testee.setup(None, 'seq_one')

        batch_sz = 5
        batchin = create_mbatch(batch_sz, 128, 128)

        im_rgb = batchin[0, :, :, 0:3]
        im_rgb = numpy.cast['uint8'](im_rgb)
        img = Image.fromarray(im_rgb)
        #img.show()

        batch = self.testee.process(batchin)
        self.assertEqual(batch.shape, (batch_sz, 128, 128, 4))

        for i in range(batch_sz):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i == 0:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(title="TestFlipAdj.test_vertical %d" % i)

        batch = self.testee.accumulate(batchin)
        self.assertEqual(batch.shape, (batch_sz*2, 128, 128, 4))
        self.assertTrue(numpy.any(batch[0] != batch[batch_sz/2+1]))
        for i in range(batch.shape[0]):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i >= 0:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(title="TestFlipAdj.test_vertical %d" % i)

    def test_both(self):
        """
        Flip batches both vertically and horizontally.
        """
        self.testee.horizontal = True
        self.testee.vertical = True
        self.testee.setup(None, 'seq_one')

        batch_sz = 5
        batchin = create_mbatch(batch_sz, 128, 128)

        im_rgb = batchin[0, :, :, 0:3]
        im_rgb = numpy.cast['uint8'](im_rgb)
        img = Image.fromarray(im_rgb)
        #img.show()

        batch = self.testee.process(batchin)
        self.assertEqual(batch.shape, (batch_sz, 128, 128, 4))

        for i in range(batch_sz):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i == 0:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(title="TestFlipAdj.test_vertical %d" % i)

        batch = self.testee.accumulate(batchin)
        self.assertEqual(batch.shape, (batch_sz*4, 128, 128, 4))
        self.assertTrue(numpy.any(batch[0] != batch[batch_sz/2+1]))
        for i in range(batch.shape[0]):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i >= 0:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(title="TestFlipAdj.test_vertical %d" % i)


class TestRotationAdj(unittest.TestCase):
    """
    Tests for RotationAdj.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = RotationAdj(min_deg=-45.0, max_deg=45.0, step=15.0)

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee

    def test_transf_count(self):
        """
        Check the transf_count() for RotationAdj
        """
        instloc = RotationAdj()
        self.assertEqual(instloc.transf_count(), 7)

        self.assertRaises(AssertionError, callableObj=RotationAdj,
                          min_deg=-45.0, max_deg=45.0, step=-1.0)

        instloc = RotationAdj(min_deg=-45.0, max_deg=45.0, step=15.0)
        self.assertEqual(instloc.transf_count(), 7)

        instloc = RotationAdj(min_deg=-45.0, max_deg=59.0, step=15.0)
        self.assertEqual(instloc.transf_count(), 7)

        instloc = RotationAdj(min_deg=-45.0, max_deg=0.0, step=0.0)
        self.assertEqual(instloc.transf_count(), 1)

        instloc = RotationAdj(min_deg=45.0, max_deg=-45.0, step=15.0)
        self.assertEqual(instloc.transf_count(), 7)

    def test_process(self):
        """
        Rotate an actual batch.
        """
        self.testee.setup(None, 'seq_one')

        batch_sz = 1
        batchin = create_mbatch(batch_sz, 128, 128)
        img = numpy.cast['uint8'](batchin[0, :, :, 0:3])
        img = Image.fromarray(img)
        #img.show(title="TestRotationAdj.original")

        batch = self.testee.process(batchin)
        self.assertEqual(batch.shape, (batch_sz, 128, 128, 4))

        for i in range(batch_sz):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i < 10:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(
                    title="TestRotationAdj.test_vertical %d" % i)

        batchin = create_mbatch(batch_sz, 128, 128)
        batch = self.testee.accumulate(batchin)
        self.assertEqual(batch.shape, (batch_sz*self.testee.transf_count(),
                                       128, 128, 4))

        for i in range(batch.shape[0]):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i < 16:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(
                    title="TestRotationAdj.test_vertical %d" % i)


class TestScalePatchAdj(unittest.TestCase):
    """
    Tests for ScalePatchAdj.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = ScalePatchAdj()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee

    def test_transf_count(self):
        """
        Check the transf_count() for ScalePatchAdj
        """
        instloc = ScalePatchAdj()
        self.assertEqual(instloc.transf_count(), 10)

        instloc = ScalePatchAdj(start_factor=0.8, end_factor=0.9, step=0.1)
        self.assertEqual(instloc.transf_count(), 10)

        instloc = ScalePatchAdj(start_factor=0.9, end_factor=0.8, step=0.1)
        self.assertEqual(instloc.transf_count(), 10)

        instloc = ScalePatchAdj(start_factor=0.8, end_factor=0.9, step=0.0)
        self.assertEqual(instloc.transf_count(), 5)

        instloc = ScalePatchAdj(step=0.0, placements=['top_left', 'top_right'])
        self.assertEqual(instloc.transf_count(), 2)

        instloc = ScalePatchAdj(outsize=(64, 128),
                                step=0.0,
                                placements=['top_left', 'top_right'])
        self.assertEqual(instloc.transf_count(), 2)

    def test_process(self):
        """
        Adjust a batch.
        """
        self.testee.setup(None, 'seq_one')

        batch_sz = 5
        batch = create_mbatch(batch_sz, 128, 128)

        batch = self.testee.process(batch)
        self.assertEqual(batch.shape, (batch_sz, 128, 128, 4))

        for i in range(batch_sz):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i < 10:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(title="TestFlipAdj.test_vertical %d" % i)

        batchin = create_mbatch(batch_sz, 128, 128)
        batch = self.testee.accumulate(batchin)
        self.assertEqual(batch.shape, (batch_sz*self.testee.transf_count(),
                               128, 128, 4))

        for i in range(batch.shape[0]):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i < 10:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(
                    title="TestScalePatchAdj.test_vertical %d" % i)


class TestGcaAdj(unittest.TestCase):
    """
    Tests for GcaAdj.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = GcaAdj()

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee

    def test_transf_count(self):
        """
        Check the transf_count() for TestGcaAdj
        """
        instloc = GcaAdj(start_scale=1., end_scale=1., step_scale=0.,
                         subtract_mean=True, use_std=True,
                         start_sqrt_bias=0., end_sqrt_bias=0.,
                         step_sqrt_bias=0.)
        self.assertEqual(instloc.transf_count(), 1)

        instloc = GcaAdj(start_scale=1., end_scale=2., step_scale=1.,
                         subtract_mean=True, use_std=True,
                         start_sqrt_bias=0., end_sqrt_bias=0.,
                         step_sqrt_bias=0.)
        self.assertEqual(instloc.transf_count(), 2)

        instloc = GcaAdj(start_scale=1., end_scale=2., step_scale=1.,
                         subtract_mean=None, use_std=True,
                         start_sqrt_bias=0., end_sqrt_bias=0.,
                         step_sqrt_bias=0.)
        self.assertEqual(instloc.transf_count(), 4)

        instloc = GcaAdj(start_scale=1., end_scale=2., step_scale=1.,
                         subtract_mean=None, use_std=None,
                         start_sqrt_bias=0., end_sqrt_bias=0.,
                         step_sqrt_bias=0.)
        self.assertEqual(instloc.transf_count(), 8)

        instloc = GcaAdj(start_scale=1., end_scale=2., step_scale=1.,
                         subtract_mean=None, use_std=None,
                         start_sqrt_bias=0., end_sqrt_bias=1.,
                         step_sqrt_bias=1.)
        self.assertEqual(instloc.transf_count(), 16)

    def test_process(self):
        """
        Adjust a batch.
        """
        self.testee.setup(None, 'seq_one')

        batch_sz = 5
        batch = create_mbatch(batch_sz, 128, 128)

        batch = self.testee.process(batch)

        for i in range(batch_sz):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i < 1:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(title="TestGcaAdj.test_vertical %d" % i)

        batch = create_mbatch(batch_sz, 128, 128)
        batch = self.testee.accumulate(batch)
        self.assertEqual(batch.shape, (batch_sz*self.testee.transf_count(),
                                       128, 128, 4))

        for i in range(batch.shape[0]):
            im_rgb = batch[i, :, :, 0:3]
            im_rgb = numpy.cast['uint8'](im_rgb)
            if i < 10:
                img = Image.fromarray(im_rgb)
                if show_im: img.show(title="TestGcaAdj.test_vertical %d" % i)


class TestAdjFromString(unittest.TestCase):
    """
    Tests for adj_from_string().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        pass

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_simple(self):
        """
        Create adjustors via adj_from_string().
        """
        adj = adj_from_string('rotate')
        self.assertIsInstance(adj, RotationAdj)
        adj = adj_from_string('flip')
        self.assertIsInstance(adj, FlipAdj)
        adj = adj_from_string('scale')
        self.assertIsInstance(adj, ScalePatchAdj)
        adj = adj_from_string('gca')
        self.assertIsInstance(adj, GcaAdj)


if __name__ == '__main__':
    show_im = False
    if True:
        unittest.main()
    else:
        unittest.main(argv=['--verbose', 'TestGcaAdj'])
