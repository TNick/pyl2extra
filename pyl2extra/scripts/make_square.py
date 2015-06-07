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

import Image
import ImageOps
import os
import logging

from pyl2extra.utils.script import setup_logging, make_argument_parser


def make_image_square(image, img_sz, outfile=None, overwrite=False):
    """
    Will resize an image.
    
    Parameters
    ----------
    image : Image.Image or str
        The image to transform. If str it is interpreted to be a path.
    img_sz : int
        New size for that image.
        
    Returns
    -------
    image : Image.image
        The transformed image.
    """
    if isinstance(image, basestring):
        image = Image.open(image)
    #NEAREST, BILINEAR, BICUBIC, ANTIALIAS
    #image = image.resize((img_sz, img_sz), resample=Image.BICUBIC)

    width, height = image.size
    
    if width > height:
       delta2 = int((width - height)/2)
       image = ImageOps.expand(image, border=(0, delta2, 0, delta2))
    else:
       delta2 = int((height - width)/2)
       image = ImageOps.expand(image, border=(delta2, 0, delta2, 0))
    image = image.resize((img_sz, img_sz), resample=Image.BICUBIC)
        
    if not outfile is None:
        if image.mode == 'P':
            fbase, ext = os.path.splitext(outfile)
            if ext == '.jpg' or ext == '.jpeg':
                orig_out = outfile
                outfile = '%s.png' % fbase
                logging.info('%s renamed to %s because jpg '
                             'does not support palletes', 
                             orig_out, outfile)
            
        if os.path.isfile(outfile):
            if not overwrite:
                logging.info('%s exists; skipping', outfile)
                return image
        
        orig_out = outfile
        for ext in ('%s.png', '%s.jpg', '%s.bmp', '%s.tiff', '%s'):
            try:        
                image.save(outfile)
                break
            except (KeyError, IOError):
                fbase, old_ext = os.path.splitext(outfile)
                outfile = ext % fbase
        if ext == '%s':
            logging.error('unable to save %s', orig_out)
    return image
    
def make_dir_square(indir, img_sz, outdir=None, 
                    recursive=False, overwrite=False):
    """
    Will resize all images in a directory.
    
    Parameters
    ----------
    indir : str
        Imput directory to scan for images.
    img_sz : int
        New size for images.
    outdir : str
        Output directory. If None the replacement is done in place.
    recursive : bool
        Wether to scan subdirectories or only top level directory.
        
    Returns
    -------
    count : int
        Number of processed images.
    """
    if outdir is None:
        outdir = indir
    count = 0
    
    def do_file(image, fname):
        """
        Process a single file.
        """
        outfile = os.path.join(outdir, fname)
        outloc = os.path.split(outfile)[0]
        if not os.path.isdir(outloc):
            os.makedirs(outloc)
        try:
            make_image_square(image, img_sz=img_sz, 
                              outfile=outfile, overwrite=overwrite)
            return count + 1
        except Exception, exc:
            logging.error('Exception while processing %s: %s', 
                          fname, exc.message, exc_info=True)
            return count
    
    if recursive:
        for root, subdirs, files in os.walk(indir):
            for fname in files:
                file_path = os.path.join(root, fname)
                fname = fname[len(indir)+1:]
                try:
                    image = Image.open(file_path)
                except IOError:
                    continue
                count = do_file(image, fname)
    else:
        for fname in os.listdir(indir):
            file_path = os.path.join(indir, fname)
            if os.path.isfile(file_path):
                try:
                    image = Image.open(file_path)
                except IOError:
                    continue
                count = do_file(image, fname)
    return count
    
def cmd_resize(args):
    """
    Shows the main GUI interface
    """
    logging.debug('input: %s', args.input)
    logging.debug('output: %s', args.output)
    logging.debug('size: %d', args.size)
    logging.debug('recursive: %s', 'True' if args.recursive else 'False')
    
    if args.output is None or len(args.output) == 0:
        args.output = args.input
    
    if args.size <= 0:
        logging.error('The new size of the image(s) must be a '
                      'positive integer (%d)', args.size)
        return
    
    if os.path.isfile(args.input):
        try:
            image = Image.open(args.input)
        except IOError, exc:
            logging.error('Input file is not a valid image '
                          '(%s)', exc.message)
        if os.path.isdir(args.output):
            args.output = os.path.join(args.output,
                                       os.path.split(args.input)[1])
            
        make_image_square(image, img_sz=args.size,
                          outfile=args.output,
                          overwrite=args.overwrite)
    elif os.path.isdir(args.input):
        count = make_dir_square(indir=args.input, 
                                img_sz=args.size,
                                outdir=args.output,
                                recursive=args.recursive,
                                overwrite=args.overwrite)
        logging.info('%d files were converted', count)
    else:
        logging.error('The input must be an existing file or directory '
                      '(%s)', args.input)
        return

def main():
    """
    Module entry point.
    """

    # look at the arguments
    parser = make_argument_parser("Resize an image and make it square.")
    parser.add_argument('size', type=int,
                        help='New size of the images')
    parser.add_argument('input', type=str,
                        help='The file or directory to convert')
    parser.add_argument('output', type=str,
                        help='The file or directory that results',
                        default='')
    parser.add_argument('--recursive', '-R', type=bool,
                        help='For directories - scan subfolders',
                        default=False)
    parser.add_argument('--overwrite', type=bool,
                        help='Overwrite existing output files',
                        default=False)
    args = parser.parse_args()

    # prepare logging
    setup_logging(args)
    logging.debug("Application starting...")

    # run based on request
    cmd_resize(args)
    logging.debug("Application ended")

if __name__ == '__main__':
    main()
