#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to interact with ImageNet resources.

@author: Nicu Tofan <nicu.tofan@gmail.com>
"""

import os
import json
import argparse
import logging
import urllib2
import threading
import time
import Queue
import magic
import hashlib
from xml.dom import minidom

logger = None

# some predefined urls
URL_SYNSET_LIST = 'http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list'
URL_SYNSET_WORDS = 'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=%s'
URL_IS_A = 'http://www.image-net.org/archive/wordnet.is_a.txt'
URL_IMAGENET_WPAGE = 'http://www.image-net.org/synset?wnid=%s'
URL_HYPONIMS = 'http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=%s&full=%d'
URL_RELEASED = 'http://www.image-net.org/api/xml/structure_released.xml'
URL_REL_STATUS = 'http://www.image-net.org/api/xml/ReleaseStatus.xml'
URL_GET_IMG = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=%s'
URL_IMG_MAPPING = 'http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=%s'

# ----------------------------------------------------------------------------

def setup_logging(args):
    """
    Setup logging configuration
    """
    global logger

    # get a path for logging config
    if not args.log_cfg is None:
        log_cfg_file = args.log_cfg
    else:
        log_cfg_file = 'logging.json'

    # read logging options
    custom_settings = os.path.exists(log_cfg_file)
    if custom_settings:
        with open(log_cfg_file, 'rt') as cfgf:
            config = json.load(cfgf)
        logging.config.dictConfig(config,
                                  disable_existing_loggers=False)
    else:
        logging.basicConfig(level=logging.INFO)

    # retreive the logger
    logger = logging.getLogger()

    # Set the root logger level.
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # if the user requested a log file give her one
    if args.log:
        file_handler = logging.FileHandler(args.log)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

# ----------------------------------------------------------------------------

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Main entry point for spotally module.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--debug', '-D',
                        action='store_true',
                        help='Display any DEBUG-level log messages, '
                             'suppressed by default.')
    parser.add_argument('--log',
                        type=str,
                        help='The log file.',
                        default=None)
    parser.add_argument('--log-cfg',
                        type=str,
                        help='The log config file in json format.',
                        default=None)

    subparsers = parser.add_subparsers(help='Available sub-commands')

    parser_a = subparsers.add_parser('synsets', help='get the list of synsets available')
    parser_a.add_argument('--url', type=str,
                          help='the address used to retreive data',
                          default=URL_SYNSET_LIST)
    parser_a.set_defaults(func=cmd_synsets)

    parser_a = subparsers.add_parser('words', help='get the list of words given a synset')
    parser_a.add_argument('synset', type=str,
                          help='the synset to retreive words for')
    parser_a.add_argument('--url', type=str,
                          help='the address used to retreive data',
                          default=URL_SYNSET_WORDS)
    parser_a.set_defaults(func=cmd_words)

    parser_a = subparsers.add_parser('hypo', help='get the list of hyponym synsets given a synset')
    parser_a.add_argument('synset', type=str,
                          help='the synset to retreive hyponyms for')
    parser_a.add_argument('--full', type=bool,
                          help='get the full tree',
                          default=False)
    parser_a.add_argument('--url', type=str,
                          help='the address used to retreive data',
                          default=URL_HYPONIMS)
    parser_a.set_defaults(func=cmd_hypos)

    parser_a = subparsers.add_parser('urls', help='get the list of image urls given a synset')
    parser_a.add_argument('synset', type=str,
                          help='the synset to retreive image urls for')
    parser_a.add_argument('--url', type=str,
                          help='the address used to retreive data',
                          default=URL_IMG_MAPPING)
    parser_a.set_defaults(func=cmd_image_urls)


    parser_a = subparsers.add_parser('downl', help='download the images from their respective urls')
    parser_a.add_argument('--path', type=str,
                          help='output directory',
                          default='.')
    parser_a.add_argument('--dry_run', type=bool,
                          help='only count the files to download',
                          default=False)
    parser_a.add_argument('--url_rs', type=str,
                          help='the address used to retreive data for relative status',
                          default=URL_REL_STATUS)
    parser_a.add_argument('--url_im', type=str,
                          help='the address used to retreive data for image mapping',
                          default=URL_IMG_MAPPING)
    parser_a.set_defaults(func=cmd_download_images)


    return parser

# ----------------------------------------------------------------------------

def list_from_url(url):
    """
    Downloads a page, assumes it is text; splits it into lines.
    """
    logger.debug('retreiving data from %s', url)
    response = urllib2.urlopen(url)
    html = response.read()
    return html.split('\n')

def dense_list_from_url(url):
    """
    Downloads a page, assumes it is text; splits it into lines.

    Trims out white space at beginning and end; removes empty lines.
    """
    return [h.strip() for h in list_from_url(url) if len(h.strip()) > 0]

def xml_elem_by_path(document, path):
    """
    Gets the xml element at a given path.
    """
    elem = document.documentElement
    for pelem in path:
        elem = elem.getElementsByTagName(pelem)[0]
    return elem

def xml_from_url(url):
    """
    Downloads a page, assumes it is xml; creates a dom document.
    """
    logger.debug('retreiving data from %s', url)
    response = urllib2.urlopen(url)
    html = response.read()
    return minidom.parseString(html)

# ----------------------------------------------------------------------------

def get_synsets(url):
    """
    Downloads the list of synsets.
    """
    logger.debug('synsets from %s', url)
    return dense_list_from_url(url)

def cmd_synsets(args):
    """
    Prints the list of synsets.
    """
    response = get_synsets(args.url)
    for sset in response:
        print sset

# ----------------------------------------------------------------------------

def get_words(url, synset):
    """
    Downloads the list of words for a synset.
    """
    logger.debug('%s from %s', synset, url)
    return dense_list_from_url(url % synset)

def cmd_words(args):
    """
    Prints the list of words for a synset.
    """
    response = get_words(args.url, args.synset)
    for sset in response:
        print sset

# ----------------------------------------------------------------------------

def get_hypos(url, synset, full_tree):
    """
    Retreives hypos :)
    """
    logger.debug('%s from %s', synset, url)
    full_tree = '1' if full_tree else '0'
    return dense_list_from_url(url % (synset, full_tree))

def cmd_hypos(args):
    """
    Prints hypos :)
    """
    response = get_hypos(args.url, args.synset, args.full)
    for sset in response:
        print sset

# ----------------------------------------------------------------------------

def get_image_count(url, is_url=True):
    """
    Retreives a list of synsets and associated number of images.

    The sysnset entry has following attributes:
    - wnid
    - numImages
    - released
    - version
    """
    logger.debug('from %s', url)

    if is_url:
        blob = xml_from_url(url)
    else:
        with open(url, 'rt') as srcf:
            html = ''.join(srcf.readlines())
            blob = minidom.parseString(html)
    dir_entries = {}
    blob = xml_elem_by_path(blob, ('images', 'synsetInfos'))
    blob = blob.getElementsByTagName('synset')
    for k in blob:
        dir_entries[k.getAttribute('wnid')] = k.getAttribute('numImages')
    return dir_entries

# ----------------------------------------------------------------------------

def get_image_synsets(url, is_url=True):
    """
    Retreives a list of synsets that have images associated with them.
    """
    logger.debug('from %s', url)

    if is_url:
        blob = xml_from_url(url)
    else:
        with open(url, 'rt') as srcf:
            html = ''.join(srcf.readlines())
            blob = minidom.parseString(html)
    blob_list = []
    blob = xml_elem_by_path(blob, ('images', 'synsetInfos'))
    blob = blob.getElementsByTagName('synset')
    for k in blob:
        blob_list.append(k.getAttribute('wnid'))
    return blob_list

# ----------------------------------------------------------------------------

def cmd_image_count(args):
    """
    Prints a list of synsets and associated number of images.
    """
    response = get_image_count(args.url, os.path.isfile(args.url))
    for k in response:
        print k, response[k]

# ----------------------------------------------------------------------------

def get_image_urls(url, synset):
    """
    Retreives a list of image names and urls for the synset.
    """
    logger.debug('%s from %s', synset, url)
    list_to_parse = dense_list_from_url(url % synset)
    result = {}
    for line in list_to_parse:
        i = line.index(' ')
        result[line[:i]] = line[i+1:]
    return result

def cmd_image_urls(args):
    """
    Prints a list of image names and urls for the synset.
    """
    response = get_image_urls(args.url, args.synset)
    for k in response:
        print k, response[k]

# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------

class ImageDownloader(object):
    """
    Class to help in downloading images.
    """
    def __init__(self, down_loc):
        """
        Constructor.
        """
        super(ImageDownloader, self).__init__()
        self.down_loc = down_loc
        self.threads_count = 10
        self.threads = []
        self.queue = Queue.Queue()
        self.finish = 0
        self.downloaded_ok = 0
        self.downloaded_fail = 0
        self.tot_img_count = 0
        self.synsets = []
        self.gen_semaphore = threading.BoundedSemaphore(self.threads_count)
        self.file_hashes = {}

    def push_synset(self, sset_name, sset_data):
        """
        Adds a synset to queue.
        """
        self.queue.put((sset_name, sset_data))

    def pop_synset(self):
        """
        Gets a synset from queue.
        """
        return self.queue.get()

    def done_synset(self, thid, sset_name, im_ok, im_fail):
        """
        A thread reports that it is done with a synset.
        """
        self.gen_semaphore.acquire()
        self.downloaded_ok = self.downloaded_ok + im_ok
        self.downloaded_fail = self.downloaded_fail + im_fail
        logging.debug('thread %d done with sset %s (%d ok, %d fail)',
                      thid, sset_name, im_ok, im_fail)
        self.gen_semaphore.release()
        self.queue.task_done()

    def thread_ended(self, thid):
        """
        Show yourself out.
        """
        logger.debug("thread %d is done", thid)
        self.gen_semaphore.acquire()
        self.finish = self.finish + 1
        self.gen_semaphore.release()

    def image_downloaded(self, full_path):
        """
        We're informed that an image was downloaded.

        Compute the hash of the file; if we saw this hash before, add it there.
        self.file_hashes will be used in purge_duplicates().
        """
        hashval = hashfile(full_path)
        self.gen_semaphore.acquire()
        try:
            self.file_hashes[hashval].append(full_path)
        except KeyError:
            self.file_hashes[hashval] = [full_path]
        self.gen_semaphore.release()

    def purge_duplicates(self):
        """
        Removes duplicates from download directory.
        """
        for fhash in self.file_hashes:
            flist = self.file_hashes[fhash]
            if len(flist) > 0:
                for fpath in flist:
                    os.remove(fpath)
                    logger.info("removed duplicate %s (%s)", fhash, fpath)
                self.downloaded_ok = self.downloaded_ok - len(flist)
                self.downloaded_fail = self.downloaded_fail + len(flist)


    def get_image(self, imname, imurl, magicf):
        """
        A thread downloads one image.
        """
        ext = os.path.splitext(imurl)[1].lower()
        err_msg = ''
        if len(ext) > 5:
            # filter out thinks like .php?a=b
            # .tiff is the longest that I can think of
            ext = ''
        out_file = os.path.join(self.down_loc, imname + ext)
        logger.info("download %s from %s to %s", imname, imurl, out_file)

        if os.path.isfile(out_file):
            logger.debug('the file already exists')
            return True

        try:
            urlf = urllib2.urlopen(imurl)
            with open(out_file, "wb") as local_file:
                local_file.write(urlf.read())
            b_ok = False
            if not os.path.isfile(out_file):
                err_msg = 'missing'
            elif os.path.getsize(out_file) == 0:
                err_msg = 'size 0'
            else:
                if len(ext) == 0:
                    try:
                        mmstr = magicf.from_file(mime=True)
                        mmstr = mmstr.split('/')
                        if mmstr[0] == 'image':
                            ext = '.' + mmstr[1].lower()
                            new_file = os.path.join(self.down_loc, imname + ext)
                            os.rename(out_file, new_file)
                            out_file = new_file
                    except (magic.MagicException, IndexError):
                        pass
                    if len(ext) == 0:
                        logger.debug('can not find a better extension')
                    self.image_downloaded(out_file)
                b_ok = True
            if not b_ok:
                logger.debug('failed to download %s from %s (%s)',
                             imname, imurl, err_msg)
        except Exception:
            b_ok = False
            logger.debug('failed to download %s from %s',
                         imname, imurl, exc_info=True)
        return b_ok

    def worker(myself, thid):
        """
        The thread.
        """
        magicf = magic.Magic(mime=True)
        logger.debug("thread %d starts", thid)
        while True:
            # get a synset
            sset = myself.pop_synset()
            if sset is None:
                time.sleep(1)
                sset = myself.pop_synset()
                if sset is None:
                    break
            logger.debug("thread %d downloads %s synset", thid, sset[0])
            image_dict = sset[1]
            im_ok = 0
            im_fail = 0
            for k in image_dict:
                if myself.get_image(k, image_dict[k], magicf):
                    im_ok = im_ok + 1
                else:
                    im_fail = im_fail + 1
            myself.done_synset(thid, sset[0], im_ok, im_fail)
        myself.thread_ended(thid)

    def run(self, url_rel, url_mapping):
        """
        Downloads everything.
        """
        # get the list of synsets
        self.synsets = get_image_synsets(url_rel)
        if self.threads_count > len(self.synsets):
            self.threads_count = len(self.synsets)
        if self.threads_count == 0:
            logging.error("No synsets")
            return

        # get a synset and start a thread
        for i in range(self.threads_count):
            sset = get_image_urls(url_mapping, self.synsets[i])
            self.push_synset(self.synsets[i], sset)
            thr = threading.Thread(target=ImageDownloader.worker, args=(self, i))
            thr.daemon = True
            self.threads.append(thr)
            thr.start()

        # get the rest of the synsets
        for i in range(self.threads_count, len(self.synsets)):
            sset = get_image_urls(URL_IMG_MAPPING, self.synsets[i])
            self.push_synset(self.synsets[i], sset)

        # make yourself useful
        ImageDownloader.worker(self, self.threads_count)

        # wait for all the threads to exit
        #while self.finish < self.threads_count:
        #    time.sleep(1)
        self.queue.join()
        self.purge_duplicates()

        assert self.downloaded_ok + self.downloaded_fail == self.tot_img_count
        logging.info('%d synsets were downloaded (%d images, %d ok, %d failed)',
                     len(self.synsets),
                     self.tot_img_count,
                     self.downloaded_ok,
                     self.downloaded_fail)

def count_images(url_rel, url_mapping):
    """
    How many images you should be getting.
    """
    synsets = get_image_synsets(url_rel)
    image_count = 0
    for i in synsets:
        sset = get_image_urls(url_mapping, i)
        for k in sset:
            logging.debug(k)
        image_count = image_count + len(sset)
    logging.info('Total images in dataset: %d', image_count)

def cmd_download_images(args):
    """
    Number of images.
    """
    if args.dry_run:
        count_images(args.url_rs, args.url_im)
    else:
        if not os.path.isdir(args.path):
            os.mkdir(args.path)
        downloader = ImageDownloader(args.path)
        downloader.run(args.url_rs, args.url_im)

# ----------------------------------------------------------------------------

def main():
    """
    Module entry point.
    """

    # look at the arguments
    parser = make_argument_parser()
    args = parser.parse_args()

    # prepare logging
    setup_logging(args)
    logger.debug("logging set-up")

    # run based on request
    args.func(args)

    logger.debug("script ended")

if __name__ == '__main__':
    main()
