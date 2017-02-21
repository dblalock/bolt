#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import os
import sys
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup
from setuptools import Extension


# ================================ C++ extension

import numpy

CPP_SRC_PATH = 'cpp/src'
# CPP_INCLUDE_PATH = 'cpp/src/include'

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# gather up all the source files
srcFiles = ['python/bolt/native.i']
includeDirs = [numpy_include]
# paths = [CPP_SRC_PATH, CPP_INCLUDE_PATH]
paths = [CPP_SRC_PATH]
for path in paths:
    srcDir = path
    for root, dirNames, fileNames in os.walk(srcDir):
        for dirName in dirNames:
            absPath = os.path.join(root, dirName)
            print('adding dir to path: %s' % absPath)
            globStr = "%s/*.c*" % absPath
            files = glob(globStr)
            if 'eigen/src' not in absPath:  # just include top level
                includeDirs.append(absPath)
            srcFiles += files

print("includeDirs:")
print(includeDirs)
print("srcFiles:")
print(srcFiles)

# set the compiler flags so it'll build on different platforms (feel free
# to file a  pull request with a fix if it doesn't work on yours)
# note that -march=native implies -mavx and -mavx2; Bolt requires AVX2
extra_args = ['-std=c++14',
              '-fno-rtti',
              '-stdlib=libc++',
              '-march=native',
              '-ffast-math']
if sys.platform == 'darwin':
    extra_args.append('-mmacosx-version-min=10.9')
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9 -stdlib=libc++ -framework Accelerate'

os.environ["CC"] = "g++"  # force compiling c as c++

# inplace extension module
nativeExt = Extension("_bolt",  # must match cpp header name with leading _
                      srcFiles,
                      define_macros=[('NDEBUG', '1')],
                      include_dirs=includeDirs,
                      # swig_opts=['-c++', '-modern'],
                      swig_opts=['-c++'],
                      extra_compile_args=extra_args
                      # extra_link_args=['-stdlib=libc++'],
                      )


# ================================ Python library

def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


modules = [splitext(basename(path))[0] for path in glob('python/bolt/*.py')]

packages = find_packages('python')

print("------------------------")
print("installing modules: ", modules)
print("found packages: ", packages)
print("------------------------")

setup(
    name='bolt',
    version='0.1.0',
    license='BSD',
    description='Fast vector compression and search',
    author='Davis Blalock',
    author_email='dblalock@mit.edu',
    url='https://github.com/dblalock/bolt',
    packages=packages,
    package_dir={'': 'python'},
    py_modules=modules,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MPL License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    install_requires=[
        'scons>=2.3',
        'numpy',
        'sphinx_rtd_theme'  # for docs
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    ext_modules=[
        nativeExt
    ],
)
