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
        with open(log_cfg_file, 'rt') as f:
            config = json.load(f)
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


    return parser

# ----------------------------------------------------------------------------

def list_from_url(url):
    logger.debug('retreiving data from %s', url)
    response = urllib2.urlopen(url)
    html = response.read()
    return html.split('\n')

def dense_list_from_url(url):
    return [h for h in list_from_url(url) if len(h) > 0]    
        
def xml_elem_by_path(document, path):
    elem = document.documentElement
    for p in path:
        elem = elem.getElementsByTagName(p)[0]
    return elem
    
def xml_from_url(url):
    logger.debug('retreiving data from %s', url)
    response = urllib2.urlopen(url)
    html = response.read()
    return minidom.parseString(html)

# ----------------------------------------------------------------------------

def get_synsets(url):
    logger.debug('synsets from %s', url)
    return dense_list_from_url(url)

def cmd_synsets(args):
    response = get_synsets(args.url)
    for s in response:
        print s
    
# ----------------------------------------------------------------------------

def get_words(url, synset):
    logger.debug('%s from %s', synset, url)
    return dense_list_from_url(url % synset)
    
def cmd_words(args):
    response = get_words(args.url, args.synset)
    for s in response:
        print s
    
# ----------------------------------------------------------------------------

def get_hypos(url, synset, full_tree):
    logger.debug('%s from %s', synset, url)
    full_tree = '1' if full_tree else '0'
    return dense_list_from_url(url % (synset, full_tree))
    
def cmd_hypos(args):
    response = get_hypos(args.url, args.synset, args.full)
    for s in response:
        print s
    
# ----------------------------------------------------------------------------

def get_image_count(url, is_url=True):
    logger.debug('from %s', url)
    
    if is_url:
        result = xml_from_url(url)
    else:
        with open(url, 'rt') as f:
            html = ''.join(f.readlines())
            result = minidom.parseString(html)
    dir_entries = {}
    result = xml_elem_by_path(result, ('images','synsetInfos'))
    result = result.getElementsByTagName('synset')
    for k in result:
        #            k.getAttribute('wnid')
        #            k.getAttribute('numImages')
        #            k.getAttribute('released')
        #            k.getAttribute('version')
        dir_entries[k.getAttribute('wnid')] = k.getAttribute('numImages')
    
def cmd_image_count(args):
    response = get_image_count(args.url, os.path.isfile(args.url))
    for k in response:
        print k, response[k]

# ----------------------------------------------------------------------------

def main():
    """
    Module entry point.
    """
    global logger
    
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
