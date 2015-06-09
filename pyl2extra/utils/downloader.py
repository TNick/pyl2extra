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

import hashlib
import logging
import magic
import multiprocessing
import os
import pycurl
import sys

from pyl2extra.utils.script import setup_logging, make_argument_parser

# We should ignore SIGPIPE when using pycurl.NOSIGNAL - see
# the libcurl tutorial for more info.
try:
    import signal
    from signal import SIGPIPE, SIG_IGN
    signal.signal(SIGPIPE, SIG_IGN)
except ImportError:
    pass

_LOGGER = logging.getLogger(__name__)

class Downloader(object):
    """
    Generates the content using separate processes.
    """
    def __init__(self, urls, outfiles, count=None,
                 compute_hash=True, auto_extension=False,
                 wait_timeout=1000, empty_error=True):
        assert len(urls) == len(outfiles)

        if count is None:
            count = multiprocessing.cpu_count()
            count = count - 1 if count > 1 else 1
        elif count < 0:
            raise ValueError("Number of processes must be a positive integer")

        #: the result should include a hash for the file
        self.compute_hash = compute_hash
        #: create extension for output file from the url
        self.auto_extension = auto_extension
        #: number of worker processes to use
        self.workers_count = count
        #: list of files to retreive
        self.urls = urls
        #: list of files to generate
        self.outfiles = outfiles
        #: number of seconds to wait before declaring timeout
        self.wait_timeout = wait_timeout
        #: a file of size 0 is considered to be an error
        self.empty_error = empty_error

        #: position in the list
        self.provider_offset = 0

        self.freelist = None
        self.magicf = magic.Magic(mime=True)
        self.multi = None
        self.keep_alive = True

        #: the results accumulate here
        self.results = []

        _LOGGER.debug("PycURL %s (compiled against 0x%x)",
                      pycurl.version, pycurl.COMPILE_LIBCURL_VERSION_NUM)
        super(Downloader, self).__init__()

    def setup(self, post_request=False):
        """
        Starts the processes and waits for orders.

        Parameters
        ----------
        post_request : bool, optional
            Also request the list of files to be downloaded. By default this
            parameter is ``False``, which means that the user must explicitly
            request the data using `self.push_request()` after
            the call to ``setup()``
        """
        self.multi = pycurl.CurlMulti()
        self.multi.handles = []
        for i in range(self.workers_count):
            cobj = pycurl.Curl()
            cobj.fp = None
            cobj.setopt(pycurl.FOLLOWLOCATION, 1)
            cobj.setopt(pycurl.MAXREDIRS, 5)
            cobj.setopt(pycurl.CONNECTTIMEOUT, 30)
            cobj.setopt(pycurl.TIMEOUT, self.wait_timeout)
            cobj.setopt(pycurl.NOSIGNAL, 1)
            cobj.setopt(pycurl.AUTOREFERER, 1)

            # After 15.000 files and 99 open files the process freezes
            # Try to mitigate this.
            cobj.setopt(pycurl.FORBID_REUSE, 1)

            try:
                if self.keep_alive:
                    cobj.setopt(pycurl.TCP_KEEPALIVE, 1)
                    cobj.setopt(pycurl.TCP_KEEPIDLE, 120)
                    cobj.setopt(pycurl.TCP_KEEPINTVL, 60)
                else:
                    cobj.setopt(pycurl.TCP_KEEPALIVE, 0)
            except AttributeError:
                _LOGGER.debug('no keep alive support')

            cobj.setopt(pycurl.SSL_VERIFYPEER, 0)
            cobj.setopt(pycurl.SSL_VERIFYHOST, 0)
            #cobj.setopt(pycurl.SSL_VERIFYRESULT, 0)

            self.multi.handles.append(cobj)
        self.freelist = self.multi.handles[:]
        _LOGGER.debug('downloader has been set up (%d workers)',
                      self.workers_count)

    def tear_down(self):
        """
        Terminates all components.
        """
        _LOGGER.debug('tearing down the downloader')
        for cobj in self.multi.handles:
            if cobj.fp is not None:
                cobj.fp.close()
                cobj.fp = None
            cobj.close()
        self.multi.close()

    def append(self, urls, outfiles, post_request=True):
        """
        Appends to the list of things to download.
        """
        assert len(urls) == len(outfiles)
        self.urls += urls
        self.outfiles += outfiles
        _LOGGER.debug('append %d => %d',
                      len(urls), len(self.urls))

    def get_all(self):
        """
        Wait until all the files were downloaded.
        """
        self.wait_for_data()
        return self.results

    def wait_for_data(self, count=None):
        """
        Waits for some provider to deliver its data.
        """
        if count is None:
            count = len(self.urls)
        _LOGGER.debug('waiting for %d items', count)
        num_processed = 0

        while self.provider_offset < count:
            # If there is an url to process and a free curl object, add to multi stack
            while self.provider_offset < count and len(self.freelist) > 0:
                url = self.urls[self.provider_offset]
                filename = self.outfiles[self.provider_offset]
                work_message = {'url': url,
                                'output': filename,
                                'hash': self.compute_hash,
                                'autoext': self.auto_extension,
                                'index': self.provider_offset}
                self.provider_offset = self.provider_offset + 1

                while True:
                    if self.auto_extension:
                        filename = put_ext_from_url(url, filename)[0]
                        if os.path.isfile(filename):
                            work_message['output'] = filename
                            self._file_downloaded(work_message, url, 'existing')
                            break
                    elif os.path.isfile(filename):
                        self._file_downloaded(work_message, url, 'existing')
                        break
                    # download the file
                    _LOGGER.debug('%s is enqueued for %s',
                                  url, work_message['output'])
                    cobj = self.freelist.pop()
                    cobj.fp = open(work_message['output'], "wb")
                    cobj.setopt(pycurl.URL, url)
                    cobj.setopt(pycurl.WRITEDATA, cobj.fp)
                    cobj.work_message = work_message
                    _LOGGER.debug('handle %s added to multi', str(cobj))
                    self.multi.add_handle(cobj)
                    break

            # Run the internal curl state machine for the multi stack
            while 1:
                ret, num_handles = self.multi.perform()
                _LOGGER.debug('perform: %d, %d', ret, num_handles)
                if ret != pycurl.E_CALL_MULTI_PERFORM:
                    break
            # Check for curl objects which have terminated, and add them to the freelist
            while 1:
                num_q, ok_list, err_list = self.multi.info_read()
                _LOGGER.debug('read: num_q %d, ok %d, err %d',
                              num_q, len(ok_list), len(err_list))
                for cobj in ok_list:
                    self._cobj_done(cobj)
                    self._file_downloaded(cobj.work_message,
                                          cobj.getinfo(pycurl.EFFECTIVE_URL))
                    self.freelist.append(cobj)

                for cobj, errno, errmsg in err_list:
                    self._cobj_done(cobj)
                    self._file_failed(cobj.work_message, errno, errmsg)
                    self.freelist.append(cobj)

                num_processed = num_processed + len(ok_list) + len(err_list)
                if num_q == 0:
                    break

            # Currently no more I/O is pending, could do something in the meantime
            # (display a progress bar, etc.).
            # We just call select() to sleep until some more data is available.
            self.multi.select(0.2)

    def _cobj_done(self, cobj):
        """
        Release a connection object.
        """
        cobj.fp.close()
        cobj.fp = None
        self.multi.remove_handle(cobj)
        _LOGGER.debug('handle %s removed from multi', str(cobj))

    def _file_downloaded(self, work_message, url, status='ok'):
        """
        A file was succesfully downloaded.
        """
        work_message['url'] = url
        if work_message['autoext']:
            fname = ext_decorator(self.magicf, work_message['output'])[1]
            work_message['output'] = fname
        if work_message['hash']:
            work_message['hash'] = hashfile(work_message['output'])
        work_message['size'] = os.path.getsize(work_message['output'])
        if work_message['size'] == 0:
            if self.empty_error:
                work_message['status'] = 'error'
                work_message['error'] = 'Empty file'
                os.remove(work_message['output'])
                _LOGGER.debug("%s empty file removed", work_message['output'])
            else:
                work_message['status'] = status
                _LOGGER.debug('Empty file: %s', work_message['output'])
        else:
            work_message['status'] = status
        self.results.append(work_message)
        _LOGGER.info("[%s] downloaded from %s to %s",
                     work_message['status'],
                     work_message['url'],
                     work_message['output'])

    def _file_failed(self, work_message, errno, errmsg):
        """
        A file could not be downloaded.
        """
        work_message['status'] = 'error'
        work_message['error'] = 'Error %d: %s' % (errno, errmsg)
        self.results.append(work_message)
        _LOGGER.error("failed to download from %s to %s (%d): %s",
                      work_message['url'],
                      work_message['output'],
                      errno, errmsg)

def hashfile(path, blocksize=65536):
    """
    Compute the hash of a file
    """
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()

def ext_decorator(magicf, fname):
    """
    Appends an extension based on mime type. Renames the file.
    """
    ext = ''
    try:
        mmstr = magicf.from_file(filename=fname)
        if mmstr == 'text/plain':
            ext = '.txt'
        elif mmstr == 'inode/x-empty':
            ext = ''
        elif mmstr == 'image/x-ms-bmp':
            ext = 'bmp'
        else:
            mmstr = mmstr.split('/')
            ext = '.' + mmstr[1].lower()
        new_file = '%s%s' % (fname, ext)
        os.rename(fname, new_file)
        fname = new_file
    except (magic.MagicException, IndexError):
        pass
    if len(ext) == 0:
        _LOGGER.debug('can not find a better extension')
    return ext, fname

def put_ext_from_url(furl, fname):
    """
    Append extension after cleaning it up.
    """
    ext = os.path.splitext(furl)[1].lower()
    if len(ext) > 0:
        for cchr in ('#', '?', '&'):
            try:
                i_cut = ext.index(cchr)
                ext = ext[:i_cut]
            except ValueError:
                pass
        if ext == 'php':
            ext = ''
        else:
            fname = '%s%s' % (fname, ext)

    _LOGGER.debug('guessed path is %s', fname)
    return fname, ext

def download_files(urls, outfiles=None, compute_hash=True,
                   auto_extension=False):
    """
    Downloads a bunch of files.

    Parameters
    ----------
    urls : str or list of str
        The url for the image(s) to download. It may be a single string or
        a list of strings.
    outfiles : str or list of str, optional
        The files where output is to be stored. If this is a single string
        it is considered to be the directory where files are to be saved.
        If None (default) current directory is assumed. If `outlist` is a
        list, its length should match the length of `flist`

    Returns
    -------
    result : list
        The result is a list of dictionaries, one for each entry:
        - ``output``: the file where output was saved
        - ``size``: the file where output was saved
        - ``url``: the file where output was saved
        - ``hash``: the file where output was saved
        - ``status``: the file where output was saved
        - ``error``: (only present if status is ``error``) the error message
    """
    if outfiles is None:
        outfiles = os.getcwd()

    if isinstance(outfiles, basestring):
        basedir = outfiles
        assert os.path.isdir(basedir)
        outfiles = []
        for url in urls:
            i_last_point = -1
            i_last_slash = 0
            i_ext_end = len(url)
            for i, chcrt in enumerate(url):
                if chcrt == '.':
                    i_last_point = i
                elif chcrt == '/':
                    i_last_slash = i + 1
                elif chcrt == '#':
                    i_ext_end = i
                    break
                elif chcrt == '?':
                    i_ext_end = i
                    break
            if i_last_point < i_last_slash:
                i_last_point = i_ext_end
            if auto_extension:
                fname = url[i_last_slash:i_last_point]
            else:
                if i_last_point+1 != i_ext_end:
                    fname = url[i_last_slash:i_last_point]
                    fname += url[i_last_point:i_ext_end].lower()
                else:
                    fname = url[i_last_slash:i_ext_end]
            outfiles.append(os.path.join(basedir, fname))

    dwnl = Downloader(urls=urls, outfiles=outfiles, count=None,
                      compute_hash=compute_hash,
                      auto_extension=auto_extension)
    dwnl.setup()
    result = dwnl.get_all()
    dwnl.tear_down()
    return result

if __name__ == '__main__':
    parser = make_argument_parser(description="download files")
    parser.add_argument('input', type=str,
                        help='The file to download')
    parser.add_argument('output', type=str,
                        help='The location')
    args = parser.parse_args()

    setup_logging(args, logger=_LOGGER)
    if len(sys.argv) < 2:
        _LOGGER.error('Nothing to download')
    else:
        toprint = download_files(urls=args.input, outfiles=args.output)
        for rslt in toprint:
            if rslt['status'] == 'error':
                _LOGGER.error('[%s] %s: %s',
                              rslt['status'], rslt['url'],
                              rslt['error'])
            else:
                _LOGGER.info('[%s] %s', rslt['status'], rslt['url'])
