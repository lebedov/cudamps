#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup

NAME =               'cudamps'
VERSION =            '0.1.1'
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'https://github.com/lebedov/cudamps/'
DESCRIPTION =        'Python interface to CUDA Multi-Process Service'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    install_requires = ['pycuda >= 2014.1',
                        'pytools >= 2014.3']
    if sys.version_info < (3, 0):
        install_requires.append('subprocess32')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        py_modules = ['cudamps'],
        install_requires = install_requires)
