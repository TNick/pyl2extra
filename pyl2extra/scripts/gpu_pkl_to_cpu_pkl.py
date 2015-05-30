#!/usr/bin/env python
"""
Converts a pickle file containing CudaNdarraySharedVariables into
a pickle file containing only TensorSharedVariables.

Usage:

gpu_pkl_to_cpu_pkl.py <gpu.pkl> <cpu.pkl>

Loads gpu.pkl, replaces cuda shared variables with numpy ones,
and saves to cpu.pkl.

If you create a model while using GPU and later want to unpickle it
on a machine without a GPU, you must convert it this way.

This is theano's fault, not pylearn2's. I would like to fix theano,
but don't understand the innards of theano well enough, and none of
the theano developers has been willing to help me at all with this
issue. If it annoys you that you have to do this, please help me
persuade the theano developers that this issue is worth more of their
attention.

Note: This script is also useful if you want to create a model on GPU,
save it, and then run other theano functionality on CPU later, even
if your machine has a GPU. It could be useful to modify this script
to do the reverse conversion, so you can create a model on CPU, save
it, and then run theano functions on GPU later.

Further note: this script is very hacky and imprecise. It is likely
to do things like blow away subclasses of list and dict and turn them
into plain lists and dicts. It is also liable to overlook all sorts of
theano shared variables if you have an exotic data structure stored in
the pickle. You probably want to test that the cpu pickle file can be
loaded on a machine without GPU to be sure that the script actually
found them all.
"""
from __future__ import print_function

__authors__ = ["Ian Goodfellow", "Nicu Tofan"]
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Ian Goodfellow", "Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

import logging
import os
from pylearn2.utils import serial
import types

# theano.config.device is read-only so we change the value in environment
# before importing theano
thflags = os.environ['THEANO_FLAGS']
if thflags:
    thflags = thflags + ",device=cpu"
else:
    thflags = "device=cpu"
os.environ['THEANO_FLAGS'] = thflags
import theano

from pyl2extra.utils.script import setup_logging, make_argument_parser


def convert_file(in_path, out_path):
    """
    Converts a model from a GPU machine in one that can be used on a CPU only
    machine.

    Parameters
    ----------
    in_path : str
        Input file. It must contain a pickled model.
    out_path : str
        Output file. Will contain a pickled model.
        If the file already exists it will be overwritten without warning.
    """
    model = convert(serial.load(in_path))
    serial.save(out_path, model)

def convert(model):
    """
    Converts a model from a GPU machine in one that can be used on a CPU only
    machine.

    Parameters
    ----------
    model : pylearn2.models.model.Model
        The model to convert.

    Returns
    -------
    model : pylearn2.models.model.Model
        Converted model.
    """
    # Map ids of objects we've fixed before to the fixed
    # version, so we don't clone objects when fixing
    # Can't use object itself as key because not all objects are hashable
    already_fixed = {}

    # ids of objects being fixed right now (we don't support cycles)
    currently_fixing = []

    blacklist = ["im_class", "func_closure", "co_argcount",
                 "co_cellvars", "func_code",
                 "append", "capitalize", "im_self",
                 "func_defaults", "func_name"]

    # The list is never used
    # blacklisted_keys = ["bytearray", "IndexError",
    #                    "isinstance", "copyright", "main"]

    postponed_fixes = []

    class Placeholder(object):
        """
        TODO: fill in description
        """
        def __init__(self, id_to_sub):
            self.id_to_sub = id_to_sub

    class FieldFixer(object):
        """
        TODO: fill in description
        """
        def __init__(self, obj, field, fixed_field):
            self.obj = obj
            self.field = field
            self.fixed_field = fixed_field

        def apply(self):
            """
            TODO: fill in description
            """
            obj = self.obj
            field = self.field
            fixed_field = already_fixed[self.fixed_field.id_to_sub]
            setattr(obj, field, fixed_field)

    def fix(obj, stacklevel=0):
        """
        TODO: fill in description
        """
        prefix = ''.join(['.']*stacklevel)
        oid = id(obj)
        canary_oid = oid
        logging.info('%sfixing %s', prefix, str(oid))
        if oid in already_fixed:
            return already_fixed[oid]
        if oid in currently_fixing:
            logging.info('returning placeholder for '+str(oid))
            return Placeholder(oid)
        currently_fixing.append(oid)
        if hasattr(obj, 'set_value'):
            # Base case: we found a shared variable, must convert it
            rval = theano.shared(obj.get_value())
            if hasattr(obj, 'name'):
                rval.name = obj.name
            # Sabotage its getstate so if something tries to
            # pickle it, we'll find out
            obj.__getstate__ = None
        elif obj is None:
            rval = None
        elif isinstance(obj, list):
            logging.info('%sfixing a list', prefix)
            rval = []
            for i, elem in enumerate(obj):
                logging.info('%s.fixing elem %d', prefix, i)
                fixed_elem = fix(elem, stacklevel + 2)
                if isinstance(fixed_elem, Placeholder):
                    raise NotImplementedError()
                rval.append(fixed_elem)
        elif isinstance(obj, dict):
            logging.info('%sfixing a dict', prefix)
            rval = obj
            """
            rval = {}
            for key in obj:
                if key in blacklisted_keys or (isinstance(key, str) and key.endswith('Error')):
                    logging.info(prefix + '.%s is blacklisted' % str(key))
                    rval[key] = obj[key]
                    continue
                logging.info(prefix + '.fixing key ' + str(key) + ' of type '+str(type(key)))
                fixed_key = fix(key, stacklevel + 2)
                if isinstance(fixed_key, Placeholder):
                    raise NotImplementedError()
                logging.info(prefix + '.fixing value for key '+str(key))
                fixed_value = fix(obj[key], stacklevel + 2)
                if isinstance(fixed_value, Placeholder):
                    raise NotImplementedError()
                rval[fixed_key] = fixed_value
            """
        elif isinstance(obj, tuple):
            logging.info('%sfixing a tuple', prefix)
            rval = []
            for i, elem in enumerate(obj):
                logging.info('%s.fixing elem %d', prefix, i)
                fixed_elem = fix(elem, stacklevel + 2)
                if isinstance(fixed_elem, Placeholder):
                    raise NotImplementedError()
                rval.append(fixed_elem)
            rval = tuple(rval)
        elif isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
            logging.info("%sskipping a function (can't pickle functions)",
                         prefix)
            rval = None
        elif isinstance(obj, (int, float, str)):
            rval = obj
        else:
            logging.info('%sfixing a generic object', prefix)
            field_names = dir(obj)
            for field in field_names:
                if isinstance(getattr(obj, field), types.MethodType):
                    logging.info('%s.%s is an instancemethod', prefix, field)
                    continue
                if field in blacklist or (field.startswith('__')):
                    logging.info('%s.%s is blacklisted', prefix, field)
                    continue
                logging.info('%s.fixing field %s', prefix, field)
                updated_field = fix(getattr(obj, field), stacklevel + 2)
                logging.info('%s.applying fix to field %s', prefix, field)
                if isinstance(updated_field, Placeholder):
                    postponed_fixes.append(FieldFixer(obj, field, updated_field))
                else:
                    try:
                        setattr(obj, field, updated_field)
                    except Exception:
                        logging.info("Could not set attribute %s",
                                     field, exc_info=True)
            rval = obj
        already_fixed[oid] = rval
        logging.info('%s stored fix for %s', prefix, str(oid))
        assert canary_oid == oid
        del currently_fixing[currently_fixing.index(oid)]
        return rval

    model = fix(model)
    assert len(currently_fixing) == 0
    for fixer in postponed_fixes:
        fixer.apply()
    return model

def main():
    """
    Module entry point.
    """
    # look at the arguments
    parser = make_argument_parser("Debugger for pylearn2 models.")
    parser.add_argument('input',
                        type=str,
                        help='The file to convert.',
                        default=None)
    parser.add_argument('output',
                        type=str,
                        help='The output file.',
                        default=None)
    args = parser.parse_args()

    # prepare logging
    setup_logging(args)
    logging.debug("Application starting...")

    # run based on request
    convert_file(args.input, args.output)
    logging.debug("Application ended")

if __name__ == '__main__':
    main()
