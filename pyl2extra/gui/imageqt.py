"""
The Python Imaging Library.

a simple Qt image interface.

history:
2006-06-03 fl: created

Copyright (c) 2006 by Secret Labs AB
Copyright (c) 2006 by Fredrik Lundh

The ImageQt Module is available only in PIL 1.1.6 so it was inserted here
to support previous version, too.

The code was retrieved from effbot_ on May 11'th, 2015.
License was retrieved from pythonware_ on May 11'th, 2015:

::

    By obtaining, using, and/or copying this software and/or its
    associated documentation, you agree that you have read, understood,
    and will comply with the following terms and conditions:

    Permission to use, copy, modify, and distribute this software and its
    associated documentation for any purpose and without fee is hereby granted,
    provided that the above copyright notice appears in all copies, and
    that both that copyright notice and this permission notice appear in
    supporting documentation, and that the name of Secret Labs AB or the author
    not be used in advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
    SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
    IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
    INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
    LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
    NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
    WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

:: _effbot: http://svn.effbot.org/public/stuff/sandbox/pil/ImageQt.py
:: _pythonware: http://www.pythonware.com/products/pil/license.htm:
"""

import Image

from PyQt4.QtGui import QImage, qRgb

##
# (Internal) Turns an RGB color into a Qt compatible color integer.
# We cannot use qRgb directly for this, since it returns a long
# integer, but setColorTable can only deal with integers.

def rgb(r, g, b):
    # use qRgb to pack the colors, and then turn the resulting long
    # into a negative integer with the same bitpattern.
    return (qRgb(r, g, b) & 0xffffff) - 0x1000000

##
# An PIL image wrapper for Qt.  Use the {@link ImageQt.image} attribute
# to access the <b>QImage</b> object.
#
# @param im An image object.

class ImageQt:

    ##
    # A reference to a QImage object.

    image = None

    def __init__(self, im):

        data = None
        colortable = None

        if im.mode == "1":
            format = QImage.Format_Mono
        elif im.mode == "L":
            format = QImage.Format_Indexed8
            colortable = []
            for i in range(256):
                colortable.append(rgb(i, i, i))
        elif im.mode == "P":
            format = QImage.Format_Indexed8
            colortable = []
            palette = im.getpalette()
            for i in range(0, len(palette), 3):
                colortable.append(rgb(*palette[i:i+3]))
        elif im.mode == "RGB":
            data = im.tostring("raw", "BGRX")
            format = QImage.Format_RGB32
        elif im.mode == "RGBA":
            try:
                data = im.tostring("raw", "BGRA")
            except SystemError:
                # workaround for earlier versions
                r, g, b, a = im.split()
                im = Image.merge("RGBA", (b, g, r, a))
            format = QImage.Format_ARGB32
        else:
            raise ValueError("unsupported image mode %r" % im.mode)

        # must keep a reference, or Qt will crash!
        self.data = data or im.tostring()

        self.image = QImage(self.data, im.size[0], im.size[1], format)

        if colortable:
            self.image.setColorTable(colortable)

    def __getattr__(self, attr):
        return getattr(self.image, attr)

##
# Wraps an image in a {@link ImageQt} wrapper.  You can pass in either an
# image object, or a file name, given either as a Python string or as a Qt
# string object.
#
# @param im Image object or file name.
# @return An {@link ImageQt} wrapper object.

def toimage(im):
    if hasattr(im, "toUtf8"):
        # FIXME - is this really the best way to do this?
        im = unicode(im.toUtf8(), "utf-8")
    if Image.isStringType(im):
        im = Image.open(im)
    return ImageQt(im)
