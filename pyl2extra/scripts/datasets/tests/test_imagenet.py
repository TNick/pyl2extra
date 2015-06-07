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
import unittest
from mock import patch, Mock
import os
import shutil
import tempfile
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from pyl2extra.scripts.datasets import imagenet 

TEST_SYNSETS = """
n04386664
n10731013
n03002948
n07609632
n03003091
n10562968
n07586179
n09929577
n07933530
n04136161
n03602194
n03703075
n12990597
"""

RELEASE_STATUS_SAMPLE = """<ReleaseStatus>
<releaseData>fall2011</releaseData>
<images>
<synsetInfos>
<synset wnid="n10801802" released="1" version="winter11" numImages="269"/>
<synset wnid="n10772937" released="1" version="winter11" numImages="58"/>
<synset wnid="n10028541" released="1" version="winter11" numImages="201"/>
<synset wnid="n10712374" released="1" version="winter11" numImages="175"/>
<synset wnid="n09878921" released="1" version="winter11" numImages="46"/>
<synset wnid="n10789415" released="1" version="winter11" numImages="48"/>
<synset wnid="n10370955" released="1" version="winter11" numImages="502"/>
</synsetInfos>
</images>
</ReleaseStatus>"""

GET_MAPPING_SAMPLE = """
n02109150_5962 http://1.jpg
n02109150_5969 http://2.jpg
n02109150_5976 http://3.jpg
n02109150_5981 http://4.jpg
n02109150_307 http://www.scbeacon.com/beacon_issues/03_09_18/images/Guidedog_pjh_091803.jpg
n02109150_323 http://www.braille.be/content/lig_braille/rapport_2005/img_05.jpg
"""


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestListFromUrl(unittest.TestCase):
    """
    Tests for list_from_url().
    """
    def test_simple(self, mock_urlopen):
        """
        testing list_from_url().
        """
        mok = Mock()
        mok.read.side_effect = ['resp1', 'resp1\nresp2', '', '    a    ']
        mock_urlopen.return_value = mok
        lst = imagenet.list_from_url('some_url')
        self.assertListEqual(lst, ['resp1'])
        
        lst = imagenet.list_from_url('some_url')
        self.assertListEqual(lst, ['resp1', 'resp2'])
        
        lst = imagenet.list_from_url('some_url')
        self.assertListEqual(lst, [''])
        
        lst = imagenet.list_from_url('some_url')
        self.assertListEqual(lst, ['    a    '])


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestDenseListFromUrl(unittest.TestCase):
    """
    Tests for dense_list_from_url().
    """
    def test_simple(self, mock_urlopen):
        """
        testing dense_list_from_url().
        """
        mok = Mock()
        mok.read.side_effect = ['resp1', 'resp1\nresp2', '',
                                '    ', '  a  ', '  a  \n  b  \n c  ',
                                '\n\na\n\nb\n\n  c']
        mock_urlopen.return_value = mok
        lst = imagenet.dense_list_from_url('some_url')
        self.assertListEqual(lst, ['resp1'])
        
        lst = imagenet.dense_list_from_url('some_url')
        self.assertListEqual(lst, ['resp1', 'resp2'])
        
        lst = imagenet.dense_list_from_url('some_url')
        self.assertListEqual(lst, [])
        
        lst = imagenet.dense_list_from_url('some_url')
        self.assertListEqual(lst, [])
        
        lst = imagenet.dense_list_from_url('some_url')
        self.assertListEqual(lst, ['a'])
        
        lst = imagenet.dense_list_from_url('some_url')
        self.assertListEqual(lst, ['a', 'b', 'c'])
        
        lst = imagenet.dense_list_from_url('some_url')
        self.assertListEqual(lst, ['a', 'b', 'c'])
        

class TestXmlElemByPath(unittest.TestCase):
    """
    Tests for xml_elem_by_path().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.doc = minidom.Document()
        
        root = self.doc.createElement('root')
        self.doc.appendChild(root)
        
        self.lv1 = self.doc.createElement('level1-1')
        root.appendChild(self.lv1)
        
        self.lv11 = self.doc.createElement('level2-1')
        self.lv1.appendChild(self.lv11)
                
        lv111 = self.doc.createElement('level3-1')
        self.lv11.appendChild(lv111)
        
        root.appendChild(self.doc.createElement('level1-2'))
        root.appendChild(self.doc.createElement('level1-3'))
        
        lv4 = self.doc.createElement('level1-4')
        root.appendChild(lv4)
        
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.doc

    def test_simple(self):
        """
        testing xml_elem_by_path().
        """
        elm = imagenet.xml_elem_by_path(self.doc, [])
        self.assertEqual(elm, self.doc.documentElement)
        
        self.assertRaises(IndexError, imagenet.xml_elem_by_path,
                          self.doc, ['nonexisting'])
        
        self.assertRaises(IndexError, imagenet.xml_elem_by_path,
                          self.doc, ['level1-1', 'nonexisting'])
                          
        elm = imagenet.xml_elem_by_path(self.doc, ['level1-1'])
        self.assertEqual(elm, self.lv1)      
        elm = imagenet.xml_elem_by_path(self.doc, ['level1-1', 'level2-1'])
        self.assertEqual(elm, self.lv11)
        

@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestXmlFromUrl(unittest.TestCase):
    """
    Tests for xml_from_url().
    """
    def test_simple(self, mock_urlopen):
        """
        testing xml_from_url().
        """
        mok = Mock()
        mok.read.side_effect = ['<root></root>',
                                '<root><el>test text</el></root>',
                                '', 
                                '    a    ']
        mock_urlopen.return_value = mok
        doc = imagenet.xml_from_url('some_url')
        self.assertEqual(doc.documentElement.tagName, 'root')
        
        doc = imagenet.xml_from_url('some_url')
        self.assertEqual(doc.documentElement.tagName, 'root')
        
        self.assertRaises(ExpatError, imagenet.xml_from_url, 'some_url')
        self.assertRaises(ExpatError, imagenet.xml_from_url, 'some_url')


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestGetSynsets(unittest.TestCase):
    """
    Tests for get_synsets().
    """
    def test_simple(self, mock_urlopen):
        """
        testing get_synsets().
        """
        mok = Mock()
        mok.read.side_effect = [TEST_SYNSETS]        
        mock_urlopen.return_value = mok

        lst = imagenet.get_synsets()
        self.assertListEqual(lst, ['n04386664', 'n10731013', 'n03002948',
                                   'n07609632', 'n03003091', 'n10562968', 
                                   'n07586179', 'n09929577', 'n07933530', 
                                   'n04136161', 'n03602194', 'n03703075', 
                                   'n12990597'])


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestGetWords(unittest.TestCase):
    """
    Tests for get_words().
    """
    def test_simple(self, mock_urlopen):
        """
        testing get_words().
        """
        mok = Mock()
        mok.read.side_effect = ["chickeree\nDouglas squirrel\n"
                                "Tamiasciurus douglasi"]
        mock_urlopen.return_value = mok

        lst = imagenet.get_words('some_url/%s', 'n07609632')
        self.assertListEqual(lst, ['chickeree',
                                   'Douglas squirrel',
                                   'Tamiasciurus douglasi'])


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestGetHypos(unittest.TestCase):
    """
    Tests for get_hypos().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        pass
        
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_simple(self, mock_urlopen):
        """
        testing get_hypos().
        """
        mok = Mock()
        mok.read.side_effect = [TEST_SYNSETS]
        mock_urlopen.return_value = mok

        lst = imagenet.get_hypos('some_url/%s-%s', 'n07609632', True)
        self.assertListEqual(lst, ['n04386664', 'n10731013', 'n03002948',
                                   'n07609632', 'n03003091', 'n10562968', 
                                   'n07586179', 'n09929577', 'n07933530', 
                                   'n04136161', 'n03602194', 'n03703075', 
                                   'n12990597'])


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestGetImageCount(unittest.TestCase):
    """
    Tests for get_image_count().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.sample = RELEASE_STATUS_SAMPLE
        
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_simple(self, mock_urlopen):
        """
        testing get_image_count().
        """
        mok = Mock()
        mok.read.side_effect = [self.sample]
        mock_urlopen.return_value = mok

        lst = imagenet.get_image_count('some_url', True)
        self.assertDictEqual(lst, {'n10801802': 269,
                                   'n10772937': 58,
                                   'n10028541': 201,
                                   'n10712374': 175,
                                   'n09878921': 46,
                                   'n10789415': 48, 
                                   'n10370955': 502})

@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestGetImageSynsets(unittest.TestCase):
    """
    Tests for get_image_synsets().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.sample = RELEASE_STATUS_SAMPLE
        
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_simple(self, mock_urlopen):
        """
        testing get_image_synsets().
        """
        mok = Mock()
        mok.read.side_effect = [self.sample]
        mock_urlopen.return_value = mok

        lst = imagenet.get_image_synsets('some_url', True)
        self.assertListEqual(lst, ['n10801802', 'n10772937', 'n10028541',
                                   'n10712374', 'n09878921', 'n10789415', 
                                   'n10370955'])


@patch('pyl2extra.scripts.datasets.imagenet.urllib2.urlopen')
class TestGetImageUrls(unittest.TestCase):
    """
    Tests for get_image_urls().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        pass
        
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        pass

    def test_simple(self, mock_urlopen):
        """
        testing get_image_urls().
        """
        mok = Mock()
        mok.read.side_effect = [GET_MAPPING_SAMPLE]
        mock_urlopen.return_value = mok

        lst = imagenet.get_image_urls('some_url/%s', 'n02109150')
        self.assertDictEqual(lst, {'n02109150_5962': 'http://1.jpg',
                                   'n02109150_5969': 'http://2.jpg',
                                   'n02109150_5976': 'http://3.jpg',
                                   'n02109150_5981': 'http://4.jpg',
                                   'n02109150_307': 'http://www.scbeacon.com/beacon_issues/03_09_18/images/Guidedog_pjh_091803.jpg',
                                   'n02109150_323': 'http://www.braille.be/content/lig_braille/rapport_2005/img_05.jpg'})


class TestHashFile(unittest.TestCase):
    """
    Tests for hashfile().
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.file_empty = os.path.join(self.tmp_dir, 'file_empty.txt')
        with open(self.file_empty, 'wt') as fhnd:
            fhnd.write('')
        self.file_a = os.path.join(self.tmp_dir, 'file_a.txt')
        with open(self.file_a, 'wt') as fhnd:
            fhnd.write('a')
        self.file_line = os.path.join(self.tmp_dir, 'file_line.txt')
        with open(self.file_line, 'wt') as fhnd:
            fhnd.write('abcdefghij')
        self.file_mlines = os.path.join(self.tmp_dir, 'file_mlines.txt')
        with open(self.file_mlines, 'wt') as fhnd:
            fhnd.write('abcdefghij\nabcdefghij\nabcdefghij\n')
        
    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir

    def test_simple(self):
        """
        testing hashfile().
        """
        self.assertEqual(imagenet.hashfile(self.file_empty), 
                         'd41d8cd98f00b204e9800998ecf8427e')
        self.assertEqual(imagenet.hashfile(self.file_a),
                         '0cc175b9c0f1b6a831c399e269772661')
        self.assertEqual(imagenet.hashfile(self.file_line),
                         'a925576942e94b2ef57a066101b48876')
        self.assertEqual(imagenet.hashfile(self.file_mlines),
                         'f90932f561733ea4558ada7ac7d27527')


if __name__ == '__main__':
    unittest.main()
