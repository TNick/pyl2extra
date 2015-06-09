#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to interact with ImageNet resources.

"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import csv
import hashlib
import Image
import logging
import os
import re
import urllib2
from xml.dom import minidom

from pyl2extra.utils.script import setup_logging, make_argument_parser
from pyl2extra.utils.downloader import Downloader

_LOGGER = logging.getLogger(__name__)

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

CACHE_FILE_REL_STS = 'ImageNetReleaseStatus.xml'

# a file has a name following this pattern (n04368840_5243.jpg)
# ``n`` indicates that it is a noun, ``04368840`` is an unique identifier
# and ``5243`` is the name for this particualr file.
IMG_FILE_REGEX = re.compile(r'([a-z][0-9]+)_([0-9]+)(\..+)*')

# ----------------------------------------------------------------------------

def list_from_url(url):
    """
    Downloads a page, assumes it is text; splits it into lines.

    Parameters
    ----------
    url : str
        The path towards the resource to retreive

    Returns
    -------
    list : list of str
        A list of lines.
    """
    _LOGGER.debug('retreiving data from %s', url)
    response = urllib2.urlopen(url)
    html = response.read()
    return html.split('\n')

def dense_list(listin):
    """
    Remove empty entries and strip non-empty ones.
    """
    return [h.strip() for h in listin if len(h.strip()) > 0]

def dense_list_from_url(url):
    """
    Downloads a page, assumes it is text; splits it into lines.

    Trims out white space at beginning and end; removes empty lines.

    Parameters
    ----------
    url : str
        The path towards the resource to retreive

    Returns
    -------
    list : list of str
        A list of trimmed, non-empty string or an empty list.
    """
    return dense_list(list_from_url(url))

def xml_elem_by_path(document, path):
    """
    Gets the xml element at a given path.

    Parameters
    ----------
    document : minidom.Document
        Target document where the path is to be searched.
    path : list of str
        The tag name for each level forming a list of strings. If
        multiple elements with same tag exist first one is retreived.

    Returns
    -------
    elem : minidom.Element
        The element that was found.
    """
    elem = document.documentElement
    for pelem in path:
        elem = elem.getElementsByTagName(pelem)[0]
    return elem

def xml_from_url(url):
    """
    Downloads a page, assumes it is xml; creates a dom document.

    Parameters
    ----------
    url : str
        The path towards the resource to retreive

    Returns
    -------
    elem : minidom.Document
        The document that was generated from remote string.
    """
    _LOGGER.debug('retreiving data from %s', url)
    response = urllib2.urlopen(url)
    html = response.read()
    return minidom.parseString(html)

# ----------------------------------------------------------------------------

def get_synsets(url=URL_SYNSET_LIST):
    """
    Downloads the list of synsets.

    This is just a simple wrapper around `dense_list_from_url()`.

    Parameters
    ----------
    url : str
        The path towards the resource to retreive. With no argument
        uses the default defined at module level.

    Returns
    -------
    elem : list of str
        A list of strings.
    """
    _LOGGER.debug('synsets from %s', url)
    return dense_list_from_url(url)

def cmd_synsets(args):
    """
    Prints the list of synsets.

    Simple wrapper around `get_synsets()` that prints retreived synsets,
    one on each line.
    To be used as a script command.

    Parameters
    ----------
    url : str
        The path towards the resource to retreive.
    """
    response = get_synsets(args.url)
    for sset in response:
        print sset

# ----------------------------------------------------------------------------

def get_words(url, synset):
    """
    Downloads the list of words for a synset.

    Parameters
    ----------
    url : str
        The path towards the resource to retreive. It must contain a ``%s``
        that gets replaced by ``synset``.
    synset : str
        The name of the sysnset or a list of sysnset names.

    Returns
    -------
    words : list of str or dict
        If ``synset`` is a string result is a list of strings,
        each one being a word or a word sequence. If ``synset`` is a list,
        then a dictionary is constructed, with keys being the synsets that
        were provided at input and values the words describing that syset.
        While a single synset may have multiple sentences describing it, only
        first one is returned in latter form.
    """
    if isinstance(synset, (list, tuple)):
        result = {}
        for sset in synset:
            _LOGGER.debug('%s from %s', sset, url)
            lst = dense_list_from_url(url % sset)
            if len(lst) > 0:
                result[sset] = lst[0]
            else:
                result[sset] = ''
        return result
    else:
        _LOGGER.debug('%s from %s', synset, url)
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

    Parameters
    ----------
    url : str
        The path towards the resource to retreive. It must contain a ``%s``
        that gets replaced by ``synset`` and another ``%s`` that will
        be ``0`` or ``1`` depending on ``full_tree``.
    synset : str
        The name of the sysnset.
    full_tree : bool
        Retreive the full hierarchy or just first level.

    Returns
    -------
    words : list of str
        A list of strings, each one being a synset identifier.
    """
    _LOGGER.debug('%s from %s', synset, url)
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

    Parameters
    ----------
    url : str
        The path towards the resource to retreive
    is_url : bool
        If True retreive the resource from an url using `xml_from_url()`,
        otherwise ``url`` is a path towards a file.

    Returns
    -------
    dir_entries : dict
        A dictionary with keys being synset identifiers and values the number
        of images for that identifier.
    """
    _LOGGER.debug('from %s', url)

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
        dir_entries[k.getAttribute('wnid')] = int(k.getAttribute('numImages'))
    return dir_entries

# ----------------------------------------------------------------------------

def get_image_synsets(url, is_url=True):
    """
    Retreives a list of synsets that have images associated with them.

    Parameters
    ----------
    url : str
        The path towards the resource to retreive
    is_url : bool
        If True retreive the resource from an url using `xml_from_url()`,
        otherwise ``url`` is a path towards a file.

    Returns
    -------
    blob_list : list
        A list of synset identifiers.
    """
    _LOGGER.debug('from %s', url)

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

def get_image_synsets_cached(url):
    """
    Retreives a list of synsets that have images associated with them.

    If the cached file does not exist in the current directory
    it is downloaded.
    """
    if not os.path.exists(CACHE_FILE_REL_STS):
        response = urllib2.urlopen(url)
        with open(CACHE_FILE_REL_STS, 'wt') as fhand:
            fhand.write(response.read())
    return get_image_synsets(CACHE_FILE_REL_STS, False)

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

    Parameters
    ----------
    url : str
        The path towards the resource to retreive
    synset : str
        The sysnset id.

    Returns
    -------
    result : dict
        A dictionary mapping image name to image url.
    """
    _LOGGER.debug('%s from %s', synset, url)
    list_to_parse = dense_list_from_url(url % synset)
    result = {}
    for line in list_to_parse:
        i = line.index(' ')
        result[line[:i]] = line[i+1:]
    return result

def get_image_urls_cached(url, synset, cache_file):
    """
    Retreives a list of image names and urls for the synset.

    Parameters
    ----------
    url : str
        The path towards the resource to retreive
    synset : str
        The sysnset id.
    cache_file : str
        Path to cache file.

    Returns
    -------
    result : dict
        A dictionary mapping image name to image url.
    """
    if not os.path.isfile(cache_file):
        response = urllib2.urlopen(url % synset)
        with open(cache_file, 'wt') as fhand:
            file_cont = response.read()
            fhand.write(file_cont)
    else:
        with open(cache_file, 'rt') as fhand:
            file_cont = fhand.read()
    list_to_parse = dense_list(file_cont.split('\n'))

    result = {}
    for line in list_to_parse:
        # ignore excluded entries
        if line.startswith('#'):
            continue
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
def get_images(url_rel, url_mapping):
    """
    Download images
    """
    downloader = Downloader(urls=[],
                            outfiles=[],
                            count=None,
                            compute_hash=True,
                            auto_extension=True,
                            wait_timeout=-1)
    downloader.setup()
    synsets = get_image_synsets(url_rel)
    image_count = 0
    for i in synsets:
        sset = get_image_urls(url_mapping, i)
        raise NotImplementedError()
        # TODO: Implement
        # downloader.append()
        for k in sset:
            _LOGGER.debug(k)
        image_count = image_count + len(sset)
    downloader.tear_down()
    _LOGGER.info('Total images in dataset: %d', image_count)

def count_images(url_rel, url_mapping):
    """
    How many images you should be getting.
    """
    synsets = get_image_synsets(url_rel)
    image_count = 0
    for i in synsets:
        sset = get_image_urls(url_mapping, i)
        for k in sset:
            _LOGGER.debug(k)
        image_count = image_count + len(sset)
    _LOGGER.info('Total images in dataset: %d', image_count)

def cmd_download_images(args):
    """
    Number of images.
    """
    raise NotImplementedError()
    # TODO: Implement
    if args.dry_run:
        count_images(args.url_rs, args.url_im)
    else:
        if not os.path.isdir(args.path):
            os.mkdir(args.path)
        #downloader = ImageDownloader(args.path)
        #downloader.run(args.url_rs, args.url_im)


def cmd_rem_img(args):
    """
    Remove images from synsets.
    """
    if args.path != '.':
        os.chdir(args.path)

    while True:
        try:
            lnr = raw_input('Enter file path and name: ')
            lnr = lnr.lower().strip()
            if lnr == 'quit' or lnr == 'q' or lnr == 'exit':
                break
            lnr = os.path.split(lnr)[1]
            try:
                sset = lnr[:lnr.index('_')]
            except ValueError:
                print 'Files must have synset_number.ext format'
                continue
            try:
                ftag = lnr[:lnr.index('.')]
            except ValueError:
                ftag = lnr
            sset_cache = '%s_urls.txt' % sset
            if not os.path.isfile(sset_cache):
                print sset_cache, ' does not exist'
                continue
            with open(sset_cache, 'rt') as fhand:
                file_cont = fhand.read()
            list_to_parse = dense_list(file_cont.split('\n'))
            comment_ftag = '#' + ftag
            for i, fline in enumerate(list_to_parse):
                if fline.startswith(ftag):
                    list_to_parse[i] = '#' + fline
                    with open(sset_cache, 'wt') as fhand:
                        fhand.write('\n'.join(list_to_parse))
                    continue
                elif fline.startswith(comment_ftag):
                    continue
        except EOFError:
            break


class TrackDuplicates(object):
    """
    Maintains a list of duplicates and the dictionary of hashes it
    has seen so far.
    """
    def __init__(self):
        #: the list of duplicate objects
        self.duplicates = []
        self.duplicates_hash = {}

    def inspect(self, rslt):
        """
        Checks to see if the result is a duplicate and maintains
        internal states.
        """
        if rslt['hash'] in self.duplicates_hash:
            dupl = self.duplicates_hash[rslt['hash']]
            if not rslt['output'] in self.duplicates:
                self.duplicates.append(rslt)
            if not dupl['output'] in self.duplicates:
                self.duplicates.append(dupl)
        elif rslt['size'] < 1024:
            if not rslt['output'] in self.duplicates:
                self.duplicates.append(rslt)
        else:
            self.duplicates_hash[rslt['hash']] = rslt

    def print_duplicates(self):
        """
        Print using standard ``_LOGGER.info()`` mechanism.
        """
        for dupl in self.duplicates:
            _LOGGER.warning('Duplicate hash: %s', dupl['output'])

    def __len__(self):
        """
        Number of duplicates found.
        """
        return len(self.duplicates)

def cmd_download_synset(args):
    """
    Download all images in a synset.
    """
    synsets = get_image_synsets_cached(args.url_rs)

    downloader = Downloader(urls=[],
                            outfiles=[],
                            count=64,
                            compute_hash=True,
                            auto_extension=True,
                            wait_timeout=-1)
    downloader.setup(post_request=False)
    tot_files = 0
    for sset in args.sset:
        if not sset in synsets:
            _LOGGER.warn('The %s synset was not found among known synsets',
                         sset)
        urls_cached_file = '%s_urls.txt' % sset
        links = get_image_urls_cached(args.url_im, sset,
                                      urls_cached_file)

        # have the links in two separate lists as required by Downloader
        out_files = []
        in_links = []
        for itm in links:
            out_files.append(os.path.join(args.path, itm))
            in_links.append(links[itm])
        downloader.append(urls=in_links,
                          outfiles=out_files,
                          post_request=True)
        tot_files = tot_files + len(in_links)

    results = downloader.get_all()
    downloader.tear_down()
    del downloader

    downloaded_ok = 0
    downloaded_err = 0
    trdpl = TrackDuplicates()
    for rslt in results:
        if rslt['status'] == 'error':
            _LOGGER.error('Failed to download %s from %s',
                          rslt['output'], rslt['url'])
            downloaded_err = downloaded_err + 1
        else:
            downloaded_ok = downloaded_ok + 1
            trdpl.inspect(rslt)

    trdpl.print_duplicates()

    _LOGGER.info('%d files in %d synsets, %d downloaded '
                 '(%d duplicates), %d failed.',
                 tot_files, len(args.sset), downloaded_ok,
                 len(trdpl), downloaded_err)

def segregate_files(path, recursive=False):
    """
    Divides the files in two lists: good and bad.

    Parameters
    ----------
    path : str
        The path to scan.
    recursive : bool
        Scan the subdirectories.

    Returns
    -------
    duplicates : list
    good_files : list
        List of file paths that
    synsets : list
        List of synsets that were identified
    """
    tot_files = 0
    synsets = []
    duplicates = []
    duplicates_hash = {}

    def inspect_file(fname, file_path):
        """Process a single file"""
        match = IMG_FILE_REGEX.match(fname)
        if match is None:
            return 0
        if not match.group(1) in synsets:
            synsets.append(match.group(1))
        hval = hashfile(file_path)
        if not duplicates_hash.has_key(hval):
            duplicates_hash[hval] = (fname, file_path)
            try:
                image = Image.open(file_path)
                del image
            except IOError:
                duplicates.append(file_path)
        else:
            duplicates.append(file_path)
            fprev = duplicates_hash[hval][1]
            if not fprev in duplicates:
                duplicates.append(fprev)
        return 1

    if recursive:
        for root, subdirs, files in os.walk(path):
            for fname in files:
                file_path = os.path.join(root, fname)
                tot_files = tot_files + inspect_file(fname, file_path)
    else:
        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            if os.path.isfile(file_path):
                tot_files = tot_files + inspect_file(fname, file_path)

    # find good files
    good_files = []
    for hval in duplicates_hash:
        fname, file_path = duplicates_hash[hval]
        if not file_path in duplicates:
            good_files.append(file_path)

    return duplicates, good_files, synsets

def cmd_outliers(args):
    """
    Print images that don't seem to be valid.

    This includes files that can't be loaded as an image and those that
    are images but their hash was found multiple times.
    """
    _LOGGER.debug('path: %s', args.path)
    _LOGGER.debug('recursive: %s', 'True' if args.recursive else 'False')

    duplicates, good_files, synsets = segregate_files(args.path,
                                                      args.recursive)
    tot_files = len(duplicates) + len(good_files)

    for dupl in duplicates:
        _LOGGER.info(dupl)

    _LOGGER.info('%d files inspected in %d synsets, '
                 '%d duplicates.',
                 tot_files, len(synsets), len(duplicates))

def cmd_good_files(args):
    """
    Print images that seem to be valid.

    The images may also be saved in a file.
    """
    _LOGGER.debug('path: %s', args.path)
    _LOGGER.debug('recursive: %s', 'True' if args.recursive else 'False')
    _LOGGER.debug('output: %s', args.output)

    duplicates, good_files, synsets = segregate_files(args.path,
                                                      args.recursive)

    for goodf in good_files:
        _LOGGER.info(goodf)
    if args.output:
        cached_descr = get_words(URL_SYNSET_WORDS, synsets)
        with open(args.output, 'wt') as fhand:
            spamwriter = csv.writer(fhand, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["File path", "Category", "Description"])
            for goodf in good_files:
                file_name = os.path.split(goodf)[1]
                match = IMG_FILE_REGEX.match(file_name)
                sset = match.group(1)
                description = cached_descr[sset]
                spamwriter.writerow([goodf, sset, description])

# ----------------------------------------------------------------------------

def make_arg_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = make_argument_parser("A script to interact with ImageNet APIs")

    subparsers = parser.add_subparsers(help='Available sub-commands')

    parser_a = subparsers.add_parser('synsets',
                                     help='get the list of synsets available')
    parser_a.add_argument('--url', type=str,
                          help='the address used to retreive data',
                          default=URL_SYNSET_LIST)
    parser_a.set_defaults(func=cmd_synsets)

    parser_a = subparsers.add_parser('words',
                                     help='get the list of words '
                                     'given a synset')
    parser_a.add_argument('synset', type=str,
                          help='the synset to retreive words for')
    parser_a.add_argument('--url', type=str,
                          help='the address used to retreive data',
                          default=URL_SYNSET_WORDS)
    parser_a.set_defaults(func=cmd_words)

    parser_a = subparsers.add_parser('hypo',
                                     help='get the list of hyponym '
                                     'synsets given a synset')
    parser_a.add_argument('synset', type=str,
                          help='the synset to retreive hyponyms for')
    parser_a.add_argument('--full', type=bool,
                          help='get the full tree',
                          default=False)
    parser_a.add_argument('--url', type=str,
                          help='the address used to retreive data',
                          default=URL_HYPONIMS)
    parser_a.set_defaults(func=cmd_hypos)

    parser_a = subparsers.add_parser('urls',
                                     help='get the list of image '
                                     'urls given a synset')
    parser_a.add_argument('synset', type=str,
                          help='the synset to retreive image urls for')
    parser_a.add_argument('--url', type=str,
                          help='the address used to retreive data',
                          default=URL_IMG_MAPPING)
    parser_a.set_defaults(func=cmd_image_urls)


    parser_a = subparsers.add_parser('downl',
                                     help='download the images from their '
                                     'respective urls')
    parser_a.add_argument('--path', type=str,
                          help='output directory',
                          default='.')
    parser_a.add_argument('--dry_run', type=bool,
                          help='only count the files to download',
                          default=False)
    parser_a.add_argument('--url_rs', type=str,
                          help='the address used to retreive '
                          'data for relative status',
                          default=URL_REL_STATUS)
    parser_a.add_argument('--url_im', type=str,
                          help='the address used to retreive '
                          'data for image mapping',
                          default=URL_IMG_MAPPING)
    parser_a.set_defaults(func=cmd_download_images)


    parser_a = subparsers.add_parser('dsset',
                                     help='download the images for a synset '
                                     'from their respective urls')
    parser_a.add_argument('--path', type=str,
                          help='output directory',
                          default='.')
    parser_a.add_argument('sset', type=str, nargs='*',
                          help='The synset to download')
    parser_a.add_argument('--dry_run', type=bool,
                          help='only count the files to download',
                          default=False)
    parser_a.add_argument('--url_rs', type=str,
                          help='the address used to retreive '
                          'data for relative status',
                          default=URL_REL_STATUS)
    parser_a.add_argument('--url_im', type=str,
                          help='the address used to retreive '
                          'data for image mapping',
                          default=URL_IMG_MAPPING)
    parser_a.set_defaults(func=cmd_download_synset)


    parser_a = subparsers.add_parser('bad',
                                     help='(interactive) remove images from '
                                     'the list of cached locations for '
                                     'a synset')
    parser_a.add_argument('--path', type=str,
                          help='path where cache files are saved',
                          default='.')
    parser_a.set_defaults(func=cmd_rem_img)


    parser_a = subparsers.add_parser('outliers',
                                     help='detect weird downloaded files')
    parser_a.add_argument('--path', type=str,
                          help='path where cache files are saved',
                          default='.')
    parser_a.add_argument('--recursive', '-R', type=bool,
                          help='For directories - scan subfolders',
                          default=False)
    parser_a.set_defaults(func=cmd_outliers)


    parser_a = subparsers.add_parser('valid',
                                     help='exclude weird downloaded files')
    parser_a.add_argument('--path', type=str,
                          help='path where cache files are saved',
                          default='.')
    parser.add_argument('--recursive', '-R', type=bool,
                        help='For directories - scan subfolders',
                        default=False)
    parser_a.add_argument('--output', '-O', type=str,
                          help='Generate a csv file for the files',
                          default=None)
    parser_a.set_defaults(func=cmd_good_files)


    return parser

# ----------------------------------------------------------------------------

def main():
    """
    Module entry point.
    """

    # look at the arguments
    parser = make_arg_parser()
    args = parser.parse_args()

    # prepare _LOGGER
    setup_logging(args, logger=_LOGGER)
    _LOGGER.debug("_LOGGER set-up")

    # run based on request
    args.func(args)

    _LOGGER.debug("script ended")

if __name__ == '__main__':
    main()
