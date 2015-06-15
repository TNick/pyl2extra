"""
Tests for remote utilities.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from fabric.api import run, local, settings, abort, cd, env
import functools
import os
import unittest
import shutil
import tempfile


from pyl2extra.utils.remote import Remote


class TestRemote(unittest.TestCase):
    """
    Tests for Remote class.
    """
    @functools.wraps(unittest.TestCase.setUp)
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.keyfile = os.path.join(self.tmp_dir, 'key.file')
        self.testee = Remote(host='somehost',
                             user='someuser',
                             port=79,
                             password='123456789',
                             key_file=self.keyfile)
        self.full_addr = 'someuser@somehost:79'

    @functools.wraps(unittest.TestCase.tearDown)
    def tearDown(self):
        del self.testee
        shutil.rmtree(self.tmp_dir)
        del self.tmp_dir

    def test_ad(self):
        """
        activate / deactivate
        """
        self.assertNotIn(self.full_addr, env.passwords)
        if not env.key_filename is None:
            self.assertNotIn(self.keyfile, env.key_filename)
            
        self.testee.activate()
        
        self.assertIn(self.full_addr, env.passwords)
        self.assertEqual('123456789', env.passwords[self.full_addr])
        self.assertIn(self.keyfile, env.key_filename)
        
        self.testee.deactivate()
    
        self.assertNotIn(self.full_addr, env.passwords)
        self.assertNotIn(self.keyfile, env.key_filename)


if __name__ == '__main__':
    unittest.main()
