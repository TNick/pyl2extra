"""
Code for printing values as they are being computed by Theano.

This mode is inspired by nan_guard module in pylearn2.devtools.
"""

__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import collections
import functools
import numpy
from theano.gof import OpWiseCLinker, WrapLinkerMany
from theano.compile import Mode
from theano import config

from pyl2extra.utils.npy import slice_1d

class PrintValMode(Mode):
    """
    Theano compilation Mode that allows printing values as they
    are being computed.
    """
    def __init__(self):
        def _print_node(i, node, fn):
            self.print_node(i, node, fn)

        wraps_linker = WrapLinkerMany([OpWiseCLinker(allow_gc=False)],
                                     [_print_node])
        super(PrintValMode, self).__init__(wraps_linker,
                                           optimizer=config.optimizer)

    def print_node(self, i, node, fn):
        self.enter_node(i, node)
        self.print_value(node=node, value=fn.inputs, is_input=True)
        fn()
        self.print_value(node=node, value=fn.outputs, is_input=False)
        self.exit_node(i, node)

    def enter_node(self, index, node):
        """
        Called before the inputs of the node were processe.

        Parameters
        ----------
        index : int
            The index of the node inside its parent.
        node : Apply
            The node object.
        """
        pass

    def exit_node(self, index, node):
        """
        Called after the outputs of the node were processe.

        Parameters
        ----------
        index : int
            The index of the node inside its parent.
        node : Apply
            The node object.
        """
        pass

    def print_value(self, node, value, is_input=None, level=0):
        """
        Turns a nested graph of lists/tuples/other objects
        into a list of objects.

        Parameters
        ----------
        value : object
            The value to print
        """
        if isinstance(value, (list, tuple, collections.ValuesView)):
            for elem in value:
                self.print_value(node=node, value=elem, level=level+1)
            print value
        else:
            print value

    def tear_down(self):
        """
        The user must call this method explicitly
        """

def _xml(sval):
    sval = str(sval)
    sval = sval.replace("<", "&lt;")
    sval = sval.replace(">", "&gt;")
    sval = sval.replace("\"", "&quot;")
    return sval

def _xmlop(op):
    return _xml(str(op.__class__).replace('<class \'', '').replace('\'>', ''))

class XmlValMode(PrintValMode):
    """
    Theano compilation Mode that allows saving intermediate values
    in an xml file.
    """
    def __init__(self, stream):
        self.stream = stream
        stream.write('<?xml version="1.0" ?>\n'
                     '<xmlvalmode>\n'
                     '<nodes>\n')
        super(XmlValMode, self).__init__()

    @functools.wraps(PrintValMode.enter_node)
    def enter_node(self, index, node):
        nodehdr = '<node hash="%s" name="%s"' \
                  ' op="%s" inputs="%d"' \
                  ' outputs="%d">\n' \
                  '<parents>\n'
        self.stream.write(nodehdr %
                          (str(hash(node)),
                           _xml(node),
                           _xmlop(node.op),
                           node.nin,
                           node.nout))
        for i, parent in enumerate(node.get_parents()):
            self.stream.write(' <parent id="%d" hash="%d" />\n' % 
                              (i, hash(parent)))
        self.stream.write('</parents>\n')

    @functools.wraps(PrintValMode.exit_node)
    def exit_node(self, index, node):
        self.stream.write('</node>\n')

    @functools.wraps(PrintValMode.print_value)
    def print_value(self, node, value, is_input=None, level=0):
        if is_input == True:
            self.stream.write('<inputs>\n')
            variables = node.inputs
        elif is_input == False:
            self.stream.write('<outputs>\n')
            variables = node.outputs
        else:
            variables = None

        self._print_value(node=node, value=value,
                          is_input=is_input, level=level,
                          variables=variables)

        if is_input == True:
            self.stream.write('</inputs>\n')
        elif is_input == False:
            self.stream.write('</outputs>\n')

    def _print_value(self, node, value, is_input=None,
                     level=0, variables=None):
        """
        """

        if isinstance(value, (list, tuple, collections.ValuesView)):
            if level > 0:
                
                if not variables is None and not isinstance(variables, list):
                    varstring = _xml(variables)
                    var_hash = hash(variables)
                else:
                    varstring = ""
                    var_hash = -1
                txtform = '%s<list level="%d" variable="%s" hash="%d">\n'
                self.stream.write(txtform % (' '*level,
                                             level,
                                             varstring,
                                             var_hash))

            for i, elem in enumerate(value):
                if isinstance(variables, list):
                    self._print_value(node=node, value=elem,
                                      level=level+1, variables=variables[i])
                else:
                    self._print_value(node=node, value=elem, level=level+1)

            if level > 0:
                self.stream.write('%s</list>\n' % (' '*level))
        elif isinstance(value, numpy.ndarray):
            if len(value.shape) == 0:
                if str(value.dtype).startswith('int'):
                    self.stream.write('%s<integer value="%d" dtype="%s" />\n' %
                                      (' '*level,
                                       int(value),
                                       value.dtype))
                elif str(value.dtype).startswith('complex'):
                    self.stream.write('%s<cplx r="%f" i="%f" dtype="%s" />\n' %
                                      (' '*level,
                                       float(numpy.real(value)),
                                       float(numpy.imag(value)),
                                       value.dtype))
                else:
                    self.stream.write('%s<real value="%f" dtype="%s" />\n' %
                                      (' '*level,
                                       float(value),
                                       value.dtype))
            else:
                self.stream.write('%s<ndarray shape="%s" dtype="%s">\n' %
                                  (' '*level,
                                   ','.join(str(x) for x in value.shape),
                                   value.dtype))

                for addr, part in slice_1d(value):
                    self.stream.write('%s <part address="%s">' %
                                      (' '*level,
                                       ','.join(str(x) for x in addr)))
                    self.stream.write('|'.join(str(x) for x in part))
                    self.stream.write('</part>\n')
                self.stream.write('%s</ndarray>\n' % (' '*level))
        else:
            self.stream.write('%s<value>%s</value>\n' % (' '*level,
                                                        str(value)))


    @functools.wraps(PrintValMode.tear_down)
    def tear_down(self):
        self.stream.write('</nodes>\n'
                          '</xmlvalmode>\n')


