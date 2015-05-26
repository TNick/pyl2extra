"""
Classes that generate content for ImgDataset.

These classes deal with the distributin of labour. The work may be done
online, in worker threads or in owrker processes.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from datetime import datetime
import dill
import functools
import logging
import multiprocessing
import numpy
import cProfile
import Queue
import threading
import time
import zmq

#from pyl2extra.datasets.img_dataset.dataset import ImgDataset
from pyl2extra.utils import slice_count

class Generator(object):
    """
    The class is used to generate content.
    """
    def __init__(self):

        #: associated dataset - the bound is created in setup() method
        self.dataset = None

        super(Generator, self).__init__()

    def is_inline(self):
        """
        Tell if this generator works on the same thread as the requester.

        Returns
        -------
        inline : bool
            True if the thread will block waiting for the result, False if
            the result is generated in paralel.
        """
        raise NotImplementedError()

    def setup(self, dataset):
        """
        Called by the dataset once it initialized itself.
        """
        self.dataset = dataset
        #assert isinstance(dataset, ImgDataset)

    def tear_down(self):
        """
        Called by the dataset fromits tear_down() method.
        """
        pass

    def __hash__(self):
        """
        Called by built-in function hash() and for operations on members
        of hashed collections including set, frozenset, and dict.
        """
        return hash(self.__class__.__name__)

    def get(self, source, next_index):
        """
        The get method used by the dataset to retreive batches of data.

        Parameters
        ----------
        source : touple of str
            A tuple of source identifiers (strings) to indicate the
            source for the data to retreive. The iterator will receive
            a ``data_specs`` argument consisting of ``(space, source)``.
        next_index : list or slice object
            The indexes of the examples to retreive specified either as a
            list or as a slice.

        Returns
        -------
        next_batch : tuple
            The result is a tuple of batches, one for each ``source`` that
            was requested. Each batch in the tuple should follow the
            dataspecs for the dataset.
        """
        raise NotImplementedError()

    def _prep_get(self, source, next_index):
        """
        Common opperations for a get() call.
        """
        count = slice_count(next_index)

        # prepare for iteration
        idx_features = -1
        idx_targets = -1
        result = []
        for i, src in enumerate(source):
            if src == 'features':
                idx_features = i
                result.append(numpy.zeros(shape=(count,
                                                 self.dataset.shape[0],
                                                 self.dataset.shape[1],
                                                 3)))
            elif src == 'targets':
                idx_targets = i
                result.append(numpy.zeros(shape=(count, 1), dtype='int32'))
            else:
                raise ValueError('%s implements <features> and <targets>; '
                                 '<%s> is not among these.' %
                                 (str(self.__class__.__name__), src))
        return count, result, idx_features, idx_targets

    def __getstate__(self):
        """
        Help pickle this instance.
        """
        return {'dataset': self.dataset}

    def __setstate__(self, state):
        """
        Help un-pickle this instance.
        """
        self.dataset = state['dataset']


class Basket(object):
    """
    Holds a number of processed images
    """
    def __init__(self, batch=None, classes=None):
        #: a list of processed images in the form of a numpy.ndarray
        self.batch = batch
        #: the classes that corespond to processed images
        self.classes = classes

    def __len__(self):
        """
        Get the number of processed images.
        """
        if self.batch is None:
            return 0
        else:
            return self.batch.shape[0]


class InlineGen(Generator):
    """
    Generates the content while the other parties wait on the same thread.
    """

    def __init__(self, profile=False):
        self.profile = profile
        if profile:
            self.profile_file = '/dev/shm/pyl2x-adj-' + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.profile_cnt = 1

        super(InlineGen, self).__init__()

    @functools.wraps(Generator.is_inline)
    def is_inline(self):
        return True

    @functools.wraps(Generator.is_inline)
    def get(self, source, next_index):
        if self.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        count, result, idx_features, idx_targets = self._prep_get(source,
                                                                  next_index)
        # iterate to collect data
        for i in range(count):
            fpath = self.dataset.data_provider.cnext()
            trg, categ = self.dataset.data_provider.read(fpath)
            categ = self.dataset.data_provider.categ2int(categ)
            trg = numpy.reshape(trg,
                                (1, trg.shape[0], trg.shape[1], trg.shape[2]))
            if idx_features > -1:
                trg = self.dataset.process(trg)
                result[idx_features][i, :, :, :] = trg
            if idx_targets > -1:
                result[idx_targets][i][0] = categ


        if self.profile:
            profiler.disable()
            profiler.dump_stats('%s.%d.profile' % (self.profile_file, self.profile_cnt))
            self.profile_cnt = self.profile_cnt + 1

        return tuple(result)

    def __getstate__(self):
        """
        Help pickle this instance.
        """
        return super(InlineGen, self).__getstate__()

    def __setstate__(self, state):
        """
        Help un-pickle this instance.
        """
        super(InlineGen, self).__setstate__(state)

class AsyncMixin(object):
    """
    Functionality that is common to threads and processes.
    """
    def __init__(self):
        #: list of cached batches; each list entry is a basket
        self.baskets = []
        #: number of cached images (in all baskets)
        self.cached_images = 0
        #: if the cache has fewer than this number of images request refill
        self.cache_refill_treshold = 5256
        #: number of images to retreive by each thread
        self.cache_refill_count = 128
        #: on termination counts the workers that exited
        self.finish = 0
        #: one time trigger ofr the threads to exit
        self._should_terminate = False
        #: number of workers to use
        self.workers_count = 0

        #: number of baskets to keep arraound if we don't have enough data
        self.keep_baskets = 128
        #: the list of baskets kept around
        self.baskets_backup = []

    def _get(self, source, next_index):
        """
        Get method common implementation.
        """
        count, result, idx_features, idx_targets = self._prep_get(source,
                                                                  next_index)
        assert count > 0
        logging.debug('get a batch of %d images (%d cached)',
                      count, self.cached_images)

        # where inside result array we're placing the data
        offset = 0
        while count > 0:
            self._new_or_backup(count)

            # get a basket from our list
            basket = self.get_basket()
            if basket is None:
                continue

            # copy the things in place
            to_copy = min(count, len(basket))
            if idx_features > -1:
                btc = basket.batch[0:to_copy, :, :, :]
                result[idx_features][offset:offset+to_copy, :, :, :] = btc
            if idx_targets > -1:
                btc = basket.categories[0:to_copy]
                result[idx_targets][offset:offset+to_copy, 0] = btc
            count = count - to_copy

            # the basket was larger so we have to put it back
            if len(basket) > to_copy:
                basket.batch = basket.batch[to_copy:, :, :, :]
                basket.categories = basket.categories[to_copy:]
                self.add_basket(basket)
            else:
                self.basked_done(basket)

        # make sure we're ready for next round
        refill = self.cache_refill_treshold - self.cached_images
        assert self.cache_refill_count > 0
        while refill > 0:
            self.push_request(self.cache_refill_count)
            refill = refill - self.cache_refill_count

        return tuple(result)

    def basked_done(self, basket):
        """
        A basket was received and it was extracted from queue.

        After the baskets are used they are normally discarded. If we're
        unable to provide examples fast enough the network will block
        waiting (sometimes for tens of seconds). To alleviate that, we keep
        arround the old examples and we serve them when there anre no
        new baskets.
        """
        if self.keep_baskets == 0:
            return
        assert self.keep_baskets > 0

        lkb = len(self.baskets_backup)
        if lkb >= self.keep_baskets:
            # make room for the new basket
            lkb = lkb - self.baskets_backup + 1
            self.baskets_backup = self.baskets_backup[lkb:]
        self.baskets_backup.append(basket)

    def _new_or_backup(self, count):
        """
        Replacement for `_wait_for_data()` that either gets examples from
        queue or from the backup list.
        """
        if len(self.baskets) > 0:
            return

        if len(self.baskets_backup) == 0:
            self._wait_for_data(count)
        else:
            if self._starving():
                refill = max(self.cache_refill_count, count)
                while refill > 0:
                    self.push_request(self.cache_refill_count)
                    refill = refill - self.cache_refill_count
            self.add_basket(self.baskets_backup)
            self.baskets_backup = []

    def __getstate__(self):
        """
        Help pickle this instance.
        """
        return super(AsyncMixin, self).__getstate__()

    def __setstate__(self, state):
        """
        Help un-pickle this instance.
        """
        super(AsyncMixin, self).__setstate__(state)

def _process_image(dataset, trg, categ, i, basket, basket_sz):
    """
    Process image and append it to the basket.
    """
    # process this image
    trg = numpy.reshape(trg,
                        (1, trg.shape[0],
                         trg.shape[1],
                         trg.shape[2]))
    trg = dataset.process(trg)

    # and append it to our batch
    if basket.batch is None:
        basket.batch = numpy.empty(shape=(basket_sz,
                                          trg.shape[1],
                                          trg.shape[2],
                                          trg.shape[3]),
                                   dtype=trg.dtype)
        basket.categories = numpy.empty(shape=(basket_sz),
                                        dtype='int32')

    # and we're done with this image
    basket.batch[i, :, :, :] = trg
    basket.categories[i] = categ


class ThreadedGen(Generator, AsyncMixin):
    """
    Generates the content using separate threads in same process.

    Parameters
    ----------
    count : int, optional
        The number of worker threads to use. If None, same number of threads
        as the number of cores minus one are used.
    """
    def __init__(self, count=None):

        if count is None:
            count = multiprocessing.cpu_count()
            count = count - 1 if count > 1 else 1
        elif count < 0:
            raise ValueError("Number of processes must be a positive integer")
        #: the list of active threads
        self.threads = []
        #: the queue to pass messages
        self.queue = Queue.Queue()
        #: semaphore
        self.gen_semaphore = threading.BoundedSemaphore(count)
        super(ThreadedGen, self).__init__()
        self.workers_count = count

    @functools.wraps(Generator.is_inline)
    def is_inline(self):
        return False

    @functools.wraps(Generator.setup)
    def setup(self, dataset):
        """
        Starts the threads and waits for orders.
        """
        self.dataset = dataset
        assert self.workers_count > 0
        # start threads
        for i in range(self.workers_count):
            thr = threading.Thread(target=ThreadedGen.worker,
                                   args=(self, i),
                                   name='ThreadedGen[%d]' % i)
            #thr.daemon = True
            self.threads.append(thr)
            thr.start()

    @functools.wraps(Generator.get)
    def get(self, source, next_index):
        return self._get(source, next_index)

    def _wait_for_data(self, count):
        """
        Waits for some provider to deliver its data.
        """
        timeout_count = 100
        while len(self.baskets) == 0:
            if self._starving():
                refill = max(self.cache_refill_count, count)
                while refill > 0:
                    self.push_request(self.cache_refill_count)
                    refill = refill - self.cache_refill_count
            logging.debug('main threads sleeps waiting for data (%d)',
                          timeout_count)
            # see if, instead of waiting useless here we can process some
            # images online ourselves.
            time.sleep(0.1)
            timeout_count = timeout_count - 1
            if timeout_count <= 0:
                raise RuntimeError('Timeout waiting for a thread to provide '
                                   'processed images in ThreadedGen.')

    def push_request(self, count):
        """
        Adds a request for a specified number of images to the queue.
        """
        self.queue.put((count))

    def _starving(self):
        """
        Tell if the queue is empty.
        """
        return self.queue.empty

    def pop_request(self):
        """
        Gets a request for a specified number of images from the queue.

        The method asks the data provider for file paths.
        """
        count = self.queue.get()
        result = []
        for i in range(count):
            self.gen_semaphore.acquire()
            fpath = self.dataset.data_provider.cnext()
            self.gen_semaphore.release()
            result.append(fpath)
        return result

    def add_basket(self, basket):
        """
        Appends a basket to the list.

        Also, keeps `cached_images` syncronized.
        """
        if isinstance(basket, Basket):
            basket = [basket]
        self.gen_semaphore.acquire()
        for bsk in reversed(basket):
            self.cached_images = self.cached_images + len(bsk)
            self.baskets.append(bsk)
        self.gen_semaphore.release()

    def get_basket(self):
        """
        Extracts a basket from the list.

        Also, keeps `cached_images` syncronized.
        """
        self.gen_semaphore.acquire()
        if len(self.baskets) == 0:
            result = None
        else:
            result = self.baskets.pop()
            self.cached_images = self.cached_images - len(result)
        self.gen_semaphore.release()
        return result

    def done_request(self, thid, basket):
        """
        A thread reports that it is done with a basket.
        """
        count = len(basket)
        logging.debug('thread %d done with a request of %d images',
                      thid, count)
        self.add_basket(basket)
        self.queue.task_done()

    def thread_ended(self, thid):
        """
        Show yourself out.
        """
        logging.debug("thread %d is done", thid)
        self.gen_semaphore.acquire()
        self.finish = self.finish + 1
        self.gen_semaphore.release()

    @functools.wraps(Generator.tear_down)
    def tear_down(self):
        """
        Terminates all threads.
        """
        logging.debug('ThreadedGen is being terminated; '
                      '%d items in queue '
                      '%d running threads.',
                      self.queue.qsize(), self.workers_count - self.finish)
        self._should_terminate = True
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()
        self.queue.join()
        self.queue = None
        self.gen_semaphore = None
        self.threads = None
        logging.debug('ThreadedGen was being terminated')

    @staticmethod
    def worker(myself, thid):
        """
        Thread entry point.
        """
        logging.debug("thread %d starts", thid)
        while not myself._should_terminate:
            # get next request from queue
            req = myself.pop_request()
            # nothing to do so take a nap
            if req is None:
                time.sleep(0.2)
                continue

            basket = Basket()
            basket_sz = len(req)
            logging.debug("thread %d will process %d images", thid, basket_sz)

            for i, fpath in enumerate(req):
                # read the file using data provider
                myself.gen_semaphore.acquire()
                b_ok = False
                try:
                    trg, categ = myself.dataset.data_provider.read(fpath)
                    categ = myself.dataset.data_provider.categ2int(categ)
                    b_ok = True
                except IOError, exc:
                    logging.error('Exception in worker loop: %s', str(exc))
                myself.gen_semaphore.release()

                if b_ok:
                    _process_image(myself.dataset, trg, categ,
                                   i, basket, basket_sz)

            # and we're done with this batch
            myself.done_request(thid, basket)

        myself.thread_ended(thid)


class ProcessGen(Generator, AsyncMixin):
    """
    Generates the content using separate processes.

    Parameters
    ----------
    count : int, optional
        The number of worker processes to use. If None, same number of
        processes as the number of cores minus one are used.

    Notes
    -----
    The 0MQ part of the class was heavily inspired by
    ``Python Multiprocessing with ZeroMQ`` TAO_ post.
    Some parts wre copied straight from provided code_.

    _code: https://github.com/taotetek/blog_examples/blob/master/python_multiprocessing_with_zeromq/workqueue_example.py
    _TAO: http://taotetek.net/2011/02/02/python-multiprocessing-with-zeromq/
    """
    if 0:
        RESULTS_ADDRESS = 'tcp://127.0.0.1:12460'
        CONTROL_ADDRESS = 'tcp://127.0.0.1:12461'
        VENTILATOR_ADDRESS = 'tcp://127.0.0.1:12462'
    else:
        RESULTS_ADDRESS = 'ipc:///tmp/pyl2x-procgen-results.ipc'
        CONTROL_ADDRESS = 'ipc:///tmp/pyl2x-procgen-control.ipc'
        VENTILATOR_ADDRESS = 'ipc:///tmp/pyl2x-procgen-ventilator.ipc'

    CTRL_FINISH = 'FINISHED'

    def __init__(self, count=None):

        if count is None:
            count = multiprocessing.cpu_count()
            count = count - 1 if count > 1 else 1
        elif count < 0:
            raise ValueError("Number of processes must be a positive integer")

        super(ProcessGen, self).__init__()
        self.workers_count = count
        #: number of requests send that were not fulfilled, yet
        self.outstanding_requests = 0
        #: keep various processes from returning same files
        self.provider_offset = 0
        #: maximum number of outstanding requests
        self.max_outstanding = 64
        #: number of seconds to wait before declaring timeout
        self.wait_timeout = 660
        #: number of extra images to request
        self.xcount = 16
        self.xcountcrt = 0
        #: used by receiver
        self.gen_semaphore = threading.BoundedSemaphore(count)

    @functools.wraps(Generator.is_inline)
    def is_inline(self):
        return False

    @functools.wraps(Generator.setup)
    def setup(self, dataset):
        """
        Starts the processes and waits for orders.
        """
        self.dataset = dataset
        self.outstanding_requests = 0
        self.dataset_provided = False

        # the thread used for receiving data
        self.receiverth = threading.Thread(target=ProcessGen.receiver_worker,
                                           args=(self,),
                                           name='ProcessGenReceiver')
        #thr.daemon = True
        self.receiverth.start()

        # Create a pool of workers to distribute work to
        assert self.workers_count > 0
        self.worker_pool = range(self.workers_count)
        for wrk_num in range(len(self.worker_pool)):
            multiprocessing.Process(target=worker, args=(wrk_num,)).start()

        # Initialize a zeromq context
        self.context = zmq.Context()

        # Set up a channel to receive results
        self.results_rcv = self.context.socket(zmq.PULL)
        self.results_rcv.bind(ProcessGen.RESULTS_ADDRESS)

        # Set up a channel to send control commands
        self.control_sender = self.context.socket(zmq.PUB)
        self.control_sender.bind(ProcessGen.CONTROL_ADDRESS)

        # Set up a channel to send work
        self.ventilator_send = self.context.socket(zmq.PUSH)
        self.ventilator_send.bind(ProcessGen.VENTILATOR_ADDRESS)

        # Give everything a second to spin up and connect
        time.sleep(0.5)


    @functools.wraps(Generator.tear_down)
    def tear_down(self):
        """
        Terminates all threads.
        """
        logging.debug('ProcessGen is being terminated; ')
        self._should_terminate = True
        # Signal to all workers that we are finsihed
        self.control_sender.send(dill.dumps(ProcessGen.CTRL_FINISH))
        logging.debug('ProcessGen was being terminated')

    @functools.wraps(Generator.get)
    def get(self, source, next_index):
        if not self.dataset_provided:
            # send workers a copy of the dataset
            self.control_sender.send(dill.dumps(self.dataset))
            self.dataset_provided = True
            time.sleep(0.5)
            refill = self.cache_refill_treshold
            assert self.cache_refill_count > 0
            while refill > 0:
                self.push_request(self.cache_refill_count)
                refill = refill - self.cache_refill_count
        return self._get(source, next_index)

    def _starving(self):
        """
        Tell if the queue is empty.
        """
        return self.outstanding_requests == 0
        
    def _wait_for_data(self, count):
        """
        Waits for some provider to deliver its data.
        """
        timeout_count = self.wait_timeout * 10
        while len(self.baskets) == 0:
            if self._starving():
                refill = max(self.cache_refill_count, count)
                while refill > 0:
                    self.push_request(self.cache_refill_count)
                    refill = refill - self.cache_refill_count
            #else:
            #    self.receive_all_messages()
            #    if len(self.baskets) != 0:
            #        break
            # see if, instead of waiting useless here we can process some
            # images online ourselves.
            time.sleep(0.1)
            timeout_count = timeout_count - 1
            if timeout_count <= 0:
                raise RuntimeError('Timeout waiting for a process to provide '
                                   'processed images in ProcessGen.')

    def push_request(self, count):
        """
        Adds a request for a specified number of images.

        Sends a request for a specified number of images down a zeromq "PUSH"
        connection to be processed by listening workers, in a round robin
        load balanced fashion.

        Parameters
        ----------
        count : int
            Number of images to retreive.
        """
        if self.outstanding_requests >= self.max_outstanding:
           # logging.debug('The number of outstanding requests is too '
           #               'high (%d); request for %d images ignored',
           #               self.outstanding_requests, count)
            return

        self.xcount = 16
        if self.xcountcrt >= self.xcount:
            self.xcountcrt = 0
        count = count + self.xcountcrt
        self.xcountcrt = self.xcountcrt + 1

        self.outstanding_requests = self.outstanding_requests + 1
        work_message = {'offset': self.provider_offset, 'count' : count}
        self.provider_offset = self.provider_offset + count
        self.ventilator_send.send_json(work_message)

    def receive_all_messages(self, no_block=True):
        """
        The "results_manager" function receives each result
        from multiple workers.
        """
        b_done = False
        baskets = []
        while not b_done:
            try:
                if no_block:
                    flags = zmq.NOBLOCK
                else:
                    self.results_rcv.pool(timeout=1*1000)
                    flags = 0
                basket = self.results_rcv.recv_pyobj(flags=flags)
                self.outstanding_requests = self.outstanding_requests - 1
                if len(basket) > 0:
                    logging.debug('A basket of %d examples has been '
                                  'received; %d outstanding requests, '
                                  '%d cached images',
                                  len(basket),
                                  self.outstanding_requests,
                                  self.cached_images)
                    baskets.append(basket)
                else:
                    logging.error("Empty basket received")
                assert self.outstanding_requests >= 0
            except zmq.ZMQError as exc:
                if exc.errno == zmq.EAGAIN:
                    b_done = True
                else:
                    raise
        if len(baskets) > 0:
            self.add_basket(baskets)
        #logging.debug("Received all messages; %d outstanding requests",
        #                          self.outstanding_requests)

    def add_basket(self, basket):
        """
        Appends a basket to the list.

        Also, keeps `cached_images` syncronized.
        """
        if isinstance(basket, Basket):
            basket = [basket]
        self.gen_semaphore.acquire()
        for bsk in reversed(basket):
            self.cached_images = self.cached_images + len(bsk)
            self.baskets.append(bsk)
        self.gen_semaphore.release()

    def get_basket(self):
        """
        Extracts a basket from the list.

        Also, keeps `cached_images` syncronized.
        """
        while True:
            if len(self.baskets) == 0:
                return None
            else:
                self.gen_semaphore.acquire()
                result = self.baskets.pop()
                self.gen_semaphore.release()
                if result.batch is None:
                    continue
                self.cached_images = self.cached_images - len(result)
                return result


    # The "ventilator" function generates a list of numbers from 0 to 10000, and
    #

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

# The "worker" functions listen on a zeromq PULL connection for "work"
# (numbers to be processed) from the ventilator, square those numbers,
# and send the results down another zeromq PUSH connection to the
# results manager.

def worker(wrk_num):
    """
    Worker process for `ProcessGen`.
    """
    logging.debug("worker process %d starts", wrk_num)

    # Initialize a zeromq context
    context = zmq.Context()

    # Set up a channel to receive work from the ventilator
    work_rcv = context.socket(zmq.PULL)
    work_rcv.connect(ProcessGen.VENTILATOR_ADDRESS)

    # Set up a channel to send result of work to the results reporter
    results_sender = context.socket(zmq.PUSH)
    results_sender.connect(ProcessGen.RESULTS_ADDRESS)

    # Set up a channel to receive control messages over
    control_rcv = context.socket(zmq.SUB)
    control_rcv.connect(ProcessGen.CONTROL_ADDRESS)
    control_rcv.setsockopt(zmq.SUBSCRIBE, "")

    # Set up a poller to multiplex the work receiver and control receiver channels
    poller = zmq.Poller()
    poller.register(work_rcv, zmq.POLLIN)
    poller.register(control_rcv, zmq.POLLIN)

    dataset = None

   # def pop_request(offset, count):
   #     """
   #     Gets a list of files to process
   #     """
   #     result = []
   #     count = count * (wrk_num+1)
   #     for i in range(count):
   #         fpath = dataset.data_provider.get(offset, count)
   #         result.append(fpath)
   #     return result

    # Loop and accept messages from both channels, acting accordingly
    while True:
        socks = dict(poller.poll())

        # If the message came from work_rcv channel, square the number
        # and send the answer to the results reporter
        if socks.get(work_rcv) == zmq.POLLIN and not dataset is None:
            work_message = work_rcv.recv_json()

            files = dataset.data_provider.get(work_message['offset'],
                                              work_message['count'])

            basket = Basket()
            basket_sz = len(files)
            logging.debug("worker process %d will process %d images",
                          wrk_num, basket_sz)

            for i, fpath in enumerate(files):
                b_ok = False
                try:
                    trg, categ = dataset.data_provider.read(fpath)
                    categ = dataset.data_provider.categ2int(categ)
                    b_ok = True
                except IOError, exc:
                    logging.error('Exception in worker loop: %s', str(exc))

                if b_ok:
                    _process_image(dataset, trg, categ,
                                   i, basket, basket_sz)

            if len(basket) == 0:
                logging.error("Worker %d sending empty basket", wrk_num)
            results_sender.send_pyobj(basket)

        # If the message came over the control channel, shut down the worker.
        if socks.get(control_rcv) == zmq.POLLIN:
            control_message = dill.loads(control_rcv.recv())
            if isinstance(control_message, basestring):
                if control_message == ProcessGen.CTRL_FINISH:
                    logging.info("Worker %i received FINSHED, quitting!",
                                 wrk_num)
                    break
            elif 'ImgDataset' in str(control_message.__class__):
                dataset = control_message

def gen_from_string(gen_name):
    """
    Creates a generator based on a string key.

    Parameters
    ----------
    gen_name : str
        A string identifying the type of Generator to use.

    Returns
    -------
    adj : Generator
        The instance that was constructed.
    """

    if gen_name == 'inline':
        return InlineGen()
    elif gen_name == 'threads':
        return ThreadedGen()
    elif gen_name == 'process':
        return ProcessGen()
    else:
        raise ValueError('%s is not a known Generator name' % gen_name)
