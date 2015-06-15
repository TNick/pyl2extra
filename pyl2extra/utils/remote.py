# -*- coding: utf-8 -*-
"""
Utilities for working with remote machines.

This is mostly a wrapper around `Fabric http://www.fabfile.org/`_ - a Python
(2.5-2.7) library and command-line tool for
streamlining the use of SSH for application deployment or systems
administration tasks.

"""
from __future__ import with_statement

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from fabric.api import run, local, settings, abort, cd, env
from fabric.operations import get, put
import os
import shutil
import tempfile

try:
    import pwd
except ImportError:
    import getpass
    pwd = None

def current_user():
    if pwd:
        return pwd.getpwuid(os.geteuid()).pw_name
    else:
        return getpass.getuser()


# list files recursivelly
LIST_FILES_PY = """
import os, sys
wrt = sys.stdout.write
for subdir, files, dirs in os.walk("."):
    for f in dirs + files:
        comp = os.path.join(subdir, f)
        lst = os.lstat(comp)
        wrt("%d/:/%s/:/%d/:/%d/:/%d/:/%d/:/%d/:/%d/:/%d/:/%d/:/%d\n" 
            % (lst.st_ino, comp, lst.st_mode, lst.st_dev, lst.st_nlink, 
               lst.st_uid, lst.st_gid, lst.st_size, lst.st_atime, 
               lst.st_mtime, lst.st_ctime))
"""

# list files non-recursivelly
LIST_FREC_PY = """
import os, sys
wrt = sys.stdout.write
for comp in os.listdir('.'):
    lst = os.lstat(comp)
    wrt("%d/:/%s/:/%d/:/%d/:/%d/:/%d/:/%d/:/%d/:/%d/:/%d/:/%d\n" 
        % (lst.st_ino, comp, lst.st_mode, lst.st_dev, lst.st_nlink, 
           lst.st_uid, lst.st_gid, lst.st_size, lst.st_atime, 
           lst.st_mtime, lst.st_ctime))
"""

class Stat(object):
    """
    Information about a file.
    """
    def __init__(self, fpath=None, st_mode=None, st_ino=None, st_dev=None, 
                 st_nlink=None, st_uid=None, st_gid=None, st_size=None, 
                 st_atime=None, st_mtime=None, st_ctime=None):
        self.st_ino = st_ino
        self.fpath = fpath
        self.st_mode = st_mode
        self.st_dev = st_dev
        self.st_nlink = st_nlink
        self.st_uid = st_uid
        self.st_gid = st_gid
        self.st_size = st_size
        self.st_atime = st_atime
        self.st_mtime = st_mtime
        self.st_ctime = st_ctime
        
    def from_str(self, string):
        """
        Get the information from a string.
        
        See `LIST_FILES_PY` and `LIST_FREC_PY`.
        """
        parts = string.split('/:/')
        self.st_ino = parts[0]
        self.fpath = parts[1]
        self.st_mode = parts[2]        
        self.st_dev = parts[3]
        self.st_nlink = parts[4]
        self.st_uid = parts[5]
        self.st_gid = parts[6]
        self.st_size = parts[7]
        self.st_atime = parts[8]
        self.st_mtime = parts[9]
        self.st_ctime = parts[10]
        
    @staticmethod
    def new(string):
        """
        Get the information from a string.
        """
        new_inst = Stat()
        new_inst.from_str()
        return new_inst
        
    
class Remote(object):
    """
    Class representing the remote machine.

    Parameters
    ----------
    host : str
        The address of the machine to connect to.
    user : str, optional
        The user name to use with remote machine. By default is the same
        username as current user.
    port : int, optional
        The port to connect to.
    password : str, optional
        The password to use. By default no passwrd is used.
    key_file : str, optional
        The key file.
    cache_loc : str, optional
        The location where files are downloaded and cached.
    """
    def __init__(self, host, user=None, port=None, password=None, 
                 key_file=None, cache_loc=None):

        if port is None:
            port = 22

        if cache_loc is None:
            cache_loc = tempfile.mkdtemp()

        #: The address of the machine to connect to.
        self.host = host
        #: The user name to use with remote machine.
        self.user = user
        #: The port to connect to.
        self.port = port
        #: The password to use. By default no passwrd is used.
        self.password = password
        #: The key file.
        self.key_file = key_file
        #: The location where files are downloaded and cached.
        self.cache_loc = cache_loc

        if user is None or len(user) == 0:
            user = current_user()
        #: full address in form user@host:port
        self.full_addr = '%s@%s:%d' % (user, self.host, self.port)

        #: current working directory
        self.cwd = '/home/%s' % user
        
        super(Remote, self).__init__()

    def activate(self):
        """
        Add information to Fabric environment.
        """
        user = self.user
        if user is None or len(user) == 0:
            user = current_user()
        env.passwords[self.full_addr] = self.password

        if isinstance(env.key_filename, list):
            env.key_filename.append(self.key_file)
        elif env.key_filename is None:
            env.key_filename = [self.key_file]
        else:
            assert isinstance(env.key_filename, basestring)
            env.key_filename = [env.key_filename, self.key_file]

    def deactivate(self):
        """
        Remove information from Fabric environment.
        """
        assert isinstance(env.key_filename, list)
        if self.key_file in env.key_filename:
            env.key_filename.remove(self.key_file)
        if self.full_addr in env.passwords:
            del env.passwords[self.full_addr]

    def active(self):
        """
        Tell if this instance is active or not.
        """
        return self.full_addr in env.passwords

    def __str__(self):
        """
        Stringify this instance.
        """
        return 'Remote(host=%s, port=%d, user=%s, pass=%d, key=%s)' %  (
            self.host, self.port, self.user,
            len(self.password), self.key_file)

    def get_file(self, remote_path, local_path=None, use_cache=True):
        """
        Copy a file from remote machine to local machine.
        
        Parameters
        ----------
        remote_path : str
            The path of the file on remote machine.
        local_path : str, optional
            Location on local machine where the file is to be downloaded.
            If empty the file is simply added to the cache directory.
        use_cache : str, optional
            Is the method allowed to check for that file on local 
            machine before downloading it or not.
            
        Returns
        -------
        result : bool
            True if all went well.
        """
        if not os.path.isabs(remote_path):
            remote_path = os.path.join(self.cwd, remote_path)
        cache_path = os.path.join(self.cache_loc, remote_path)
        
        if use_cache:
            if os.path.isfile(cache_path):
                if local_path is None:
                    return True
                shutil.copy(cache_path, local_path)
                return True
        
        result = get(remote_path=remote_path, local_path=cache_path)
        if result.succeeded and not local_path is None:
            shutil.copy(cache_path, local_path)
        return result.succeeded

    def put_file(self, remote_path, local_path, use_cache=True):
        """
        Copy a file from local machine to remote machine.
        
        Parameters
        ----------
        remote_path : str
            The path of the file on remote machine.
        local_path : str, optional
            Location on local machine where the file is located.
        use_cache : str, optional
            The file is to be also copied to local cache.
            
        Returns
        -------
        result : bool
            True if all went well.
        """
        if not os.path.isabs(remote_path):
            remote_path = os.path.join(self.cwd, remote_path)
        cache_path = os.path.join(self.cache_loc, remote_path)
        
        if use_cache:
            shutil.copy(local_path, cache_path)
        
        result = put(remote_path=remote_path, local_path=cache_path, 
                     mirror_local_mode=True)
        return result.succeeded

    def list_dir(self, remote_path, recursive=False):
        """
        Get the content of a directory.
        
        Parameters
        ----------
        remote_path : str
            The path of the directory on remote machine.
        recursive : bool, optional
            Only files in this directory or include subdirectories.
            
        Returns
        -------
        result : list
            Each element is a Stat instance with information about the file.
        """
        if not os.path.isabs(remote_path):
            remote_path = os.path.join(self.cwd, remote_path)
        
        with cd(remote_path):
            if 0:
                string_ = run("for i in *; do echo $i; done")
                files = string_.replace("\r","").split("\n")
            elif 0:
                string_ = run("ls -la")
                files = string_.replace("\r","").split("\n")
            else:
                if recursive:
                    string_ = run("echo " + LIST_FREC_PY + " | python")
                else:
                    string_ = run("echo " + LIST_FILES_PY + " | python")
                string_ = string_.replace("\r","").split("\n")
                files = [Stat.new(i) for i in string_]
            return files


    def test_connection(self):
        """
        See if we can connect to this remote machine.

        Returns
        -------
        result : bool
            The test command was executed.
        """
        if not self.active():
            return False
        if self.host is None or len(self.host) == 0:
            return False
        if self.user is None or len(self.user) == 0:
            return False
        return run('uname') == 'Linux'


env.connection_attempts = 1
env.timeout = 1
env.skip_bad_hosts = False
env.warn_only = True
#env.password = ''

