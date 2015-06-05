#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Examples:

    # simply run the program
    python viewprep.py

    # run run the app in debug mode
    python viewprep.py --debug

"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"


import functools
import unittest
from mock import patch, Mock
import os
import shutil
import tempfile

from pyl2extra.utils.downloader import Downloader, download_files

URL_TO_TEST = ["http://1.txt",
               "http://some-site/2.txt",
               "http://s/o/m/e/s/i/t/e/3.TIFF",
               "http://4.png?param=value",
               "http://a/b/c/5.jpg?p1=v1,p2=v2",
               "http://6.html#anchor"]
EXPECTED_OUT = ["1.txt",
                "2.txt",
                "3.tiff",
                "4.png",
                "5.jpg",
                "6.html"]

def prepare_mock(mock_urlopen):
    """
    Prepare mocking url
    """
    mok = Mock()
    mok.read.side_effect = URL_TO_TEST
    mock_urlopen.return_value = mok


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestDownloader(unittest.TestCase):
    """
    Tests for Downloader().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.testee = None
        self.tmp_dir = tempfile.mkdtemp()
        self.urls = URL_TO_TEST
        self.output = [os.path.join(self.tmp_dir, i) for i in EXPECTED_OUT]

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        self.testee.tear_down()
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir
        del self.testee

    def test_constructor(self, mock_urlopen):
        """
        testing constructor for initial state of the instance.
        """
        prepare_mock(mock_urlopen)
        self.testee = Downloader(urls=self.urls, outfiles=self.output)

        self.assertGreaterEqual(self.testee.workers_count, 1)
        self.assertEqual(self.testee.outstanding_requests, 0)
        self.assertEqual(self.testee.provider_offset, 0)
        self.assertGreaterEqual(self.testee.max_outstanding, 1)
        self.assertGreaterEqual(self.testee.wait_timeout, 1)
        self.assertNotEqual(self.testee.gen_semaphore, None)
        self.assertEqual(self.testee._should_terminate, False)
        self.assertGreaterEqual(self.testee.workers_count, 1)
        self.assertListEqual(self.testee.urls, self.urls)
        self.assertListEqual(self.testee.urls, self.urls)

        self.assertEqual(self.testee.context, None)
        self.assertEqual(self.testee.results_rcv, None)
        self.assertEqual(self.testee.control_sender, None)
        self.assertEqual(self.testee.ventilator_send, None)

    def test_simple(self, mock_urlopen):
        """
        testing get_all().
        """
        prepare_mock(mock_urlopen)
        self.testee = Downloader(urls=self.urls, outfiles=self.output)
        self.testee.setup()

        self.assertNotEqual(self.testee.context, None)
        self.assertNotEqual(self.testee.results_rcv, None)
        self.assertNotEqual(self.testee.control_sender, None)
        self.assertNotEqual(self.testee.ventilator_send, None)
        self.assertTrue(self.testee.starving())

        result = self.testee.get_all()
        for fname in self.output:
            self.assertTrue(os.path.isfile(fname))
        for rslt in result:
            self.assertTrue(rslt['hash'])

    def test_ext(self, mock_urlopen):
        """
        testing extracting extensions().
        """
        prepare_mock(mock_urlopen)

        output = [os.path.splitext(i)[0] for i in self.output]

        self.testee = Downloader(urls=self.urls, outfiles=output,
                                 compute_hash=False, auto_extension=True)
        self.testee.setup()
        result = self.testee.get_all()
        for i, url in enumerate(self.urls):
            rslt = None
            for rcandidate in result:
                if rcandidate['url'] == url:
                    rslt = rcandidate
            output = self.output[i]
            self.assertEqual(rslt['url'], url)
            self.assertEqual(rslt['output'], output)
            self.assertFalse(rslt['hash'])
            self.assertTrue(rslt['autoext'])
            self.assertEqual(rslt['status'], 'ok')


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestFunction(unittest.TestCase):
    """
    Tests for download_files().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.urls = URL_TO_TEST
        self.output = [os.path.join(self.tmp_dir, i) for i in EXPECTED_OUT]

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir

    def test_out_none(self, mock_urlopen):
        """
        testing download_files.
        """
        prepare_mock(mock_urlopen)

        os.chdir(self.tmp_dir)
        result = download_files(self.urls,
                                outfiles=None,
                                compute_hash=True,
                                auto_extension=False)
        for fname in self.output:
            self.assertTrue(os.path.isfile(fname))
        for rslt in result:
            self.assertTrue(rslt['hash'])

    def test_out_dir(self, mock_urlopen):
        """
        testing download_files.
        """
        prepare_mock(mock_urlopen)

        result = download_files(self.urls,
                                outfiles=self.tmp_dir,
                                compute_hash=True,
                                auto_extension=False)
        for fname in self.output:
            self.assertTrue(os.path.isfile(fname))
        for rslt in result:
            self.assertTrue(rslt['hash'])


    def test_out_list(self, mock_urlopen):
        """
        testing download_files.
        """
        prepare_mock(mock_urlopen)

        result = download_files(self.urls,
                                outfiles=self.output,
                                compute_hash=True,
                                auto_extension=False)
        for fname in self.output:
            self.assertTrue(os.path.isfile(fname))
        for rslt in result:
            self.assertTrue(rslt['hash'])

if __name__ == '__main__':
    unittest.main()
