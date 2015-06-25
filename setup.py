#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is the centre of all activity in building, distributing,
and installing using the Distutils. It was inspired by
(setup.py <https://github.com/lisa-lab/pylearn2/blob/master/setup.py>)_
in pylearn2 library.
"""

from __future__ import print_function

from setuptools import setup, find_packages
from setuptools.command.install import install

import os
import sys
import subprocess

if sys.version_info > (3, 0):
    ask_user = input
else:
    ask_user = raw_input

cmdclass = {}
ext_modules = []
base_path = os.path.split(os.path.abspath(__file__))[0]


class Pyl2ExtraInstall(install):
    """
    Customize install process.
    """
    def run(self):
        try:
            skip_pylearn2 = bool(os.environ['PYL2EXTRA_SKIP_PYL2'])
        except KeyError:
            skip_pylearn2 = False

        if os.path.isdir(os.path.join(base_path, 'pylearn2')):
            if not skip_pylearn2:
                crt_dir = os.getcwd()
                os.chdir(os.path.join(base_path, 'pylearn2'))
                subprocess.call(['python', 'setup.py', 'install'])
                os.chdir(crt_dir)

        # allow for unattended install (pylearn2 still needs input)
        try:
            mode = os.environ['PYL2EXTRA_INSTALL_MODE']
        except KeyError:
            mode = None

        while mode not in ['', 'install', 'develop', 'cancel']:
            if mode is not None:
                print("Please try again")
            mode = ask_user("pyl2extra installation mode: [develop]/install/cancel: ")
        if mode in ['', 'develop']:
            self.distribution.run_command('develop')
        if mode == 'install':
            return install.run(self)


cmdclass.update({'install': Pyl2ExtraInstall})

setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    name='pyl2extra',
    version='0.1dev',
    packages=find_packages(),
    description='Extends pylearn2 library.',
    license='BSD 3-clause license',
    long_description=open('ReadMe.md', 'rb').read().decode('utf8'),
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano'],
    install_requires=['numpy>=1.5', 'pyyaml', 'argparse', "Theano",
                      'python-magic', 'webcolors', 'dill', 'pillow',
                      'appdirs'],
    scripts=[],
    package_data={
        '': ['*.cu', '*.cuh', '*.h'],
    },
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering']
)
