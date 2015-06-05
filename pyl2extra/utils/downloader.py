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

import dill
import hashlib
import logging
import magic
import multiprocessing
import os
import sys
import threading
import time
import urllib2
import zmq


class Downloader(object):
    """
    Generates the content using separate processes.

    Parameters
    ----------
    urls : list of str
        The urls for the images to download.
    outlist : list of str
        The files where output is to be stored; its length should
        match the length of `flist`
    count : int, optional
        The number of worker processes to use. If None, same number of
        processes as the number of cores minus one are used.
    compute_hash : bool, optional
        Wether a hash for the output file is to be computed and stored
        in the results.
    auto_extension : bool, optional
        Wether the extension should be determined from input file. If ``True``
        the elements in ``outlist`` must not contain extensions.

    Notes
    -----
    The 0MQ part of the class was heavily inspired by
    ``Python Multiprocessing with ZeroMQ`` TAO_ post.
    Some parts wre copied straight from provided code_.

    _code: https://github.com/taotetek/blog_examples/blob/master/python_multiprocessing_with_zeromq/workqueue_example.py
    _TAO: http://taotetek.net/2011/02/02/python-multiprocessing-with-zeromq/
    """
    if 0:
        RESULTS_ADDRESS = 'tcp://127.0.0.1:12470'
        CONTROL_ADDRESS = 'tcp://127.0.0.1:12471'
        VENTILATOR_ADDRESS = 'tcp://127.0.0.1:12472'
    else:
        RESULTS_ADDRESS = 'ipc:///tmp/pyl2x-downloader-results.ipc'
        CONTROL_ADDRESS = 'ipc:///tmp/pyl2x-downloader-control.ipc'
        VENTILATOR_ADDRESS = 'ipc:///tmp/pyl2x-downloader-ventilator.ipc'

    CTRL_FINISH = 'FINISHED'

    def __init__(self, urls, outfiles, count=None,
                 compute_hash=True, auto_extension=False):

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
        #: number of requests send that were not fulfilled, yet
        self.outstanding_requests = 0
        #: keep various processes from working on same files
        self.provider_offset = 0
        #: maximum number of outstanding requests
        self.max_outstanding = 64
        #: number of seconds to wait before declaring timeout
        self.wait_timeout = 660
        #: used by receiver
        self.gen_semaphore = threading.BoundedSemaphore(count)
        #: on termination counts the workers that exited
        self.finish = 0
        #: one time trigger for the threads to exit
        self._should_terminate = False
        #: list of files to retreive
        self.urls = urls
        #: list of files to generate
        self.outfiles = outfiles
        #: the thread used for receiving data
        self.receiverth = None

        #: a zeromq context
        self.context = None
        #: a channel to receive results
        self.results_rcv = None
        #: a channel to send control commands
        self.control_sender = None
        #: a channel to send work
        self.ventilator_send = None

        #: the results accumulate here
        self.results = []

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
        self.outstanding_requests = 0
        self.receiverth = threading.Thread(target=Downloader.receiver_worker,
                                           args=(self,),
                                           name='DownloaderReceiver')
        #thr.daemon = True
        self.receiverth.start()

        # Create a pool of workers to distribute work to
        assert self.workers_count > 0
        for wrk_num in range(self.workers_count):
            multiprocessing.Process(target=_worker, args=(wrk_num,)).start()

        # Initialize a zeromq context
        self.context = zmq.Context()

        # Set up a channel to receive results
        self.results_rcv = self.context.socket(zmq.PULL)
        self.results_rcv.bind(Downloader.RESULTS_ADDRESS)

        # Set up a channel to send control commands
        self.control_sender = self.context.socket(zmq.PUB)
        self.control_sender.bind(Downloader.CONTROL_ADDRESS)

        # Set up a channel to send work
        self.ventilator_send = self.context.socket(zmq.PUSH)
        self.ventilator_send.bind(Downloader.VENTILATOR_ADDRESS)

        # Give everything a second to spin up and connect
        time.sleep(0.5)

        if post_request:
            self.push_request()

    def tear_down(self):
        """
        Terminates all components.
        """
        logging.debug('Downloader is being terminated; ')
        self._should_terminate = True
        # Signal to all workers that we are finsihed
        if not self.control_sender is None:
            self.control_sender.send(dill.dumps(Downloader.CTRL_FINISH))
        logging.debug('Downloader was being terminated')

        # Give everything a second to spin down
        time.sleep(0.5)

    def append(self, urls, outfiles, post_request=True):
        """
        Appends to the list of things to download.
        """
        assert len(urls) == len(outfiles)
        self.urls.append(urls)
        self.outfiles.append(outfiles)
        if post_request:
            self.push_request(len(urls))

    def starving(self):
        """
        Tell if the queue is empty.
        """
        return self.outstanding_requests == 0

    def get_all(self):
        """
        Wait until all the files were downloaded.
        """
        count = len(self.urls)
        self.push_request(count)
        self._wait_for_data(count)
        return self.results

    def _wait_for_data(self, count):
        """
        Waits for some provider to deliver its data.
        """
        timeout_count = self.wait_timeout * 10
        while count > len(self.results):
            if self.starving():
                self.push_request(count)
            time.sleep(0.1)
            timeout_count = timeout_count - 1
            if timeout_count <= 0:
                raise RuntimeError('Timeout waiting for a process to provide '
                                   'processed images in Downloader.')

    def push_request(self, count=None):
        """
        Adds a request for a specified number of images.

        Sends a request for a specified number of images down a zeromq "PUSH"
        connection to be processed by listening workers, in a round robin
        load balanced fashion.

        Parameters
        ----------
        count : int, optional
            Number of images to retreive. Default is to request all images
            in the list.
        """
        if count is None:
            count = len(self.urls)
        for i in range(count):
            #if self.outstanding_requests >= self.max_outstanding:
            #    return
            if self.provider_offset + 1 > len(self.urls):
                return
            self.outstanding_requests = self.outstanding_requests + 1
            work_message = {'url': self.urls[self.provider_offset],
                            'output': self.outfiles[self.provider_offset],
                            'hash': self.compute_hash,
                            'autoext': self.auto_extension}
            self.provider_offset = self.provider_offset + 1
            self.ventilator_send.send_json(work_message)

    def receive_all_messages(self, no_block=True):
        """
        The "results_manager" function receives each result
        from multiple workers.
        """
        b_done = False
        while not b_done:
            try:
                if no_block:
                    flags = zmq.NOBLOCK
                else:
                    self.results_rcv.pool(timeout=1*1000)
                    flags = 0
                result = self.results_rcv.recv_json(flags=flags)
                logging.info(result)
                self.outstanding_requests = self.outstanding_requests - 1
                self.results.append(result)
                assert self.outstanding_requests >= 0
            except zmq.ZMQError as exc:
                if exc.errno == zmq.EAGAIN:
                    b_done = True
                else:
                    raise

    @staticmethod
    def receiver_worker(myself):
        """
        Thread entry point.
        """
        logging.debug("worker thread starts")
        time.sleep(0.5)
        while not myself._should_terminate:
            myself.receive_all_messages(no_block=True)
            time.sleep(0.01)


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


# The "worker" functions listen on a zeromq PULL connection for "work"
# (numbers to be processed) from the ventilator, square those numbers,
# and send the results down another zeromq PUSH connection to the
# results manager.

def _worker(wrk_num):
    """
    Worker process for `Downloader`.
    """
    logging.debug("worker process %d starts", wrk_num)

    # Initialize a zeromq context
    context = zmq.Context()

    # Set up a channel to receive work from the ventilator
    work_rcv = context.socket(zmq.PULL)
    work_rcv.connect(Downloader.VENTILATOR_ADDRESS)

    # Set up a channel to send result of work to the results reporter
    results_sender = context.socket(zmq.PUSH)
    results_sender.connect(Downloader.RESULTS_ADDRESS)

    # Set up a channel to receive control messages over
    control_rcv = context.socket(zmq.SUB)
    control_rcv.connect(Downloader.CONTROL_ADDRESS)
    control_rcv.setsockopt(zmq.SUBSCRIBE, "")

    # Set up a poller to multiplex the work receiver and control receiver channels
    poller = zmq.Poller()
    poller.register(work_rcv, zmq.POLLIN)
    poller.register(control_rcv, zmq.POLLIN)

    magicf = magic.Magic(mime=True)

    # Loop and accept messages from both channels, acting accordingly
    while True:
        socks = dict(poller.poll())

        # If the message came from work_rcv channel, square the number
        # and send the answer to the results reporter
        if socks.get(work_rcv) == zmq.POLLIN:
            work_message = work_rcv.recv_json()
            try:

                furl = work_message['url']
                fname = work_message['output']
                compute_hash = work_message['hash']
                autoext = work_message['autoext']

                exc = ''
                if autoext:
                    exc = os.path.splitext(furl)[1].lower()
                    if len(exc) > 0:
                        try:
                            i_cut = exc.index('#')
                            exc = exc[:i_cut]
                        except ValueError:
                            pass
                        try:
                            i_cut = exc.index('?')
                            exc = exc[:i_cut]
                        except ValueError:
                            pass
                        fname = '%s%s' % (fname, exc)

                logging.info("download from %s to %s", furl, fname)

                if os.path.isfile(fname):
                    logging.debug('the file already exists')
                    work_message['status'] = 'existing'
                else:
                    urlf = urllib2.urlopen(furl)
                    with open(fname, "wb") as local_file:
                        local_file.write(urlf.read())
                    if compute_hash:
                        work_message['hash'] = hashfile(fname)
                    if len(exc) == 0 and autoext:
                        try:
                            mmstr = magicf.from_file(filename=local_file,
                                                     mime=True)
                            mmstr = mmstr.split('/')
                            ext = '.' + mmstr[1].lower()
                            new_file = '%s.%s' % (fname, exc)
                            os.rename(fname, new_file)
                            fname = new_file
                        except (magic.MagicException, IndexError):
                            pass
                        if len(ext) == 0:
                            logging.debug('can not find a better extension')

                work_message['output'] = fname
                work_message['size'] = os.path.getsize(fname)
                work_message['status'] = 'ok'
            except Exception, exc:
                work_message['status'] = 'error'
                work_message['error'] = '%s: %s' % (str(exc.__class__),
                                                    exc.message)
            results_sender.send_json(work_message)

        # If the message came over the control channel, shut down the worker.
        if socks.get(control_rcv) == zmq.POLLIN:
            control_message = dill.loads(control_rcv.recv())
            if isinstance(control_message, basestring):
                if control_message == Downloader.CTRL_FINISH:
                    logging.info("Worker %i received FINSHED, quitting!",
                                 wrk_num)
                    break

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
    if len(sys.argv) < 2:
        logging.error('Nothing to download')
    else:
        toprint = download_files(sys.argv[1:])
        for rslt in toprint:
            if rslt['status'] == 'error':
                logging.error('[%s] %s: %s',
                              rslt['status'], rslt['url'],
                              rslt['error'])
            else:
                logging.info('[%s] %s', rslt['status'], rslt['url'])
