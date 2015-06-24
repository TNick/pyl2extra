"""
Tests for images module.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import csv
import functools
import os
import shutil
import tempfile
import unittest

os.environ['THEANO_FLAGS'] = 'optimizer=fast_compile,' \
    'exception_verbosity=high,' \
    'device=cpu,' \
    'floatX=float64'

from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.costs.cost import MethodCost
from pylearn2.models.mlp import Softmax
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import MLP, ConvRectifiedLinear
from pylearn2.models.mlp import RectifiedLinear
from pylearn2.models.mlp import LinearGaussian
from pylearn2.models.mlp import mean_of_targets
from pylearn2.models.mlp import beta_from_targets
from pylearn2.training_algorithms.bgd import BGD

from pyl2extra.datasets.images import Images
from pyl2extra.testing import images


def create_csvs(tmp_dir, testset):
    """
    Create the .csv files.
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


class TestGeneric(unittest.TestCase):
    """
    Tests for Images.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.class_cnt = 8
        self.testee = None
        self.tmp_dir = tempfile.mkdtemp()
        self.testset = images.create(self.tmp_dir, 10)
        self.csv_file_fc, self.csv_file_f = create_csvs(self.tmp_dir,
                                                        self.testset)

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.testee
        del self.testset

    def test_csv_files_categ(self):
        """
        Test a csv file that has both categories and files.
        """
        self.testee = Images(source=self.csv_file_fc,
                             image_size=None, classes=self.class_cnt)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertTupleEqual(self.testee.y.shape, (len(self.testset), 1))
        self.assertTrue(self.testee.has_targets())

    def test_csv_files(self):
        """
        Test a csv file that has only files.
        """
        self.testee = Images(source=self.csv_file_f,
                             image_size=None, classes=self.class_cnt)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertIsNone(self.testee.y)
        self.assertFalse(self.testee.has_targets())

    def test_dict_categs(self):
        """
        Test with a dictionary of file paths and categories.
        """
        tst = {}
        for fpath in self.testset:
            tst[fpath] = self.testset[fpath][2]
        self.testee = Images(source=tst,
                             image_size=79, classes=self.class_cnt)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertTupleEqual(self.testee.y.shape, (len(self.testset), 1))
        self.assertTrue(self.testee.has_targets())

    def test_dict_no_categs(self):
        """
        Test with a dictionary of file paths and no categories.
        """
        tst = {}
        for fpath in self.testset:
            tst[fpath] = None
        self.testee = Images(source=tst,
                             image_size=79, classes=0)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertIsNone(self.testee.y)
        self.assertFalse(self.testee.has_targets())

    def test_dict_images(self):
        """
        Test with a dictionary of images and categories.
        """
        tst = {}
        for fpath in self.testset:
            tst[self.testset[fpath][1]] = self.testset[fpath][2]
        self.testee = Images(source=tst,
                             image_size=79, classes=self.class_cnt)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertTupleEqual(self.testee.y.shape, (len(self.testset), 1))
        self.assertTrue(self.testee.has_targets())

    def test_list_images(self):
        """
        Test with a list of image paths and no categories.
        """
        lst_img = []
        for fpath in self.testset:
            lst_img.append(fpath)
        self.testee = Images(source=[lst_img],
                             image_size=79, classes=self.class_cnt)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertIsNone(self.testee.y)
        self.assertFalse(self.testee.has_targets())

    def test_list_images_and_categs(self):
        """
        Test with a list of image paths and categories.
        """
        lst_img = []
        lst_categ = []
        for fpath in self.testset:
            lst_img.append(fpath)
            lst_categ.append(self.testset[fpath][2])
        self.testee = Images(source=(lst_img, lst_categ),
                             image_size=79, classes=self.class_cnt)
        self.assertEqual(len(self.testset), self.testee.get_num_examples())
        self.assertTupleEqual(self.testee.y.shape, (len(self.testset), 1))
        self.assertTrue(self.testee.has_targets())


class TestClassification(unittest.TestCase):
    """
    Tests for the dataset.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.class_cnt = 8
        self.image_size = 64
        self.tmp_dir = tempfile.mkdtemp()
        self.images = images.create(self.tmp_dir)
        self.csv_file_fc, self.csv_file_f = create_csvs(self.tmp_dir,
                                                        self.images)
        self.dataset = Images(source=self.csv_file_fc,
                              image_size=self.image_size,
                              classes=self.class_cnt)
        self.algorithm = SGD(
            batch_size=2,
            learning_rate=.17,
            learning_rule=Momentum(init_momentum=.5),
            monitoring_dataset={'train': self.dataset},
            cost=MethodCost(method='cost_from_X'),
            termination_criterion=EpochCounter(max_epochs=2))
        layer_0 = ConvRectifiedLinear(layer_name='h0',
                                      output_channels=64,
                                      kernel_shape=(2, 2),
                                      kernel_stride=(1, 1),
                                      pool_shape=(2, 2),
                                      pool_stride=(1, 1),
                                      irange=0.005,
                                      border_mode='valid',
                                      init_bias=0.,
                                      left_slope=0.0,
                                      max_kernel_norm=1.935,
                                      pool_type='max',
                                      tied_b=True,
                                      monitor_style="classification")
        layer_y = Softmax(max_col_norm=1.9365,
                          layer_name='y',
                          binary_target_dim=1,
                          n_classes=self.class_cnt,
                          irange=.005)
        window_shape = [self.image_size, self.image_size]
        input_space = Conv2DSpace(shape=window_shape,
                                  num_channels=3,
                                  axes=['b', 0, 1, 'c'])
        model = MLP(layers=[layer_0, layer_y],
                    input_space=input_space)
        self.train_obj = Train(dataset=self.dataset,
                               model=model,
                               algorithm=self.algorithm,
                               save_path=None,
                               save_freq=0,
                               extensions=[])

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir

    def test_file(self):
        """
        Loading the parameters for small network.
        """
        self.train_obj.main_loop()
        self.assertTrue(self.train_obj.allow_overwrite)
        self.assertTrue(self.train_obj.exceeded_time_budget)
        self.assertGreater(self.train_obj.total_seconds.eval(), 0)
        self.assertGreater(self.train_obj.training_seconds.eval(), 0)


class TestRegression(unittest.TestCase):
    """
    Tests for the dataset.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):

        self.image_size = 64
        self.tmp_dir = tempfile.mkdtemp()
        self.images = images.create(self.tmp_dir)
        self.csv_file_fc, self.csv_file_f = create_csvs(self.tmp_dir,
                                                        self.images)
        self.dataset = Images(source=self.csv_file_fc,
                              image_size=self.image_size,
                              classes=None)
        layer_0 = RectifiedLinear(layer_name='h0',
                                  dim=1200,
                                  irange=.05,
                                  max_col_norm=1.9365)
        layer_1 = RectifiedLinear(layer_name='h1',
                                  dim=1200,
                                  irange=.05,
                                  max_col_norm=1.9365)
        layer_2 = LinearGaussian(init_bias=mean_of_targets(self.dataset),
                                 init_beta=beta_from_targets(self.dataset),
                                 min_beta=1.,
                                 max_beta=100.,
                                 beta_lr_scale=1.,
                                 dim=1,
                                 layer_name='y',
                                 irange=.005)
        model = MLP(layers=[layer_0, layer_1, layer_2],
                    nvis=self.image_size*self.image_size*3)
        self.algorithm = BGD(line_search_mode='exhaustive',
                             batch_size=1024,
                             conjugate=1,
                             reset_conjugate=0,
                             reset_alpha=0,
                             updates_per_batch=10,
                             monitoring_dataset={'train': self.dataset},
                             termination_criterion=EpochCounter(max_epochs=2))
        self.train_obj = Train(dataset=self.dataset,
                               model=model,
                               algorithm=self.algorithm,
                               save_path=None,
                               save_freq=0,
                               extensions=[])

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir

    def test_file(self):
        """
        Loading the parameters for small network.
        """
        self.train_obj.main_loop()
        self.assertTrue(self.train_obj.allow_overwrite)
        self.assertTrue(self.train_obj.exceeded_time_budget)
        self.assertGreater(self.train_obj.total_seconds.eval(), 0)
        self.assertGreater(self.train_obj.training_seconds.eval(), 0)


if __name__ == '__main__':
    if True:
        unittest.main()
    else:
        unittest.main(argv=['--verbose', 'TestRegression'])
