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
from setuptools.command.install import install


# ================================ C++ extension

import numpy

PROJ_DIR = os.path.dirname(os.path.realpath(__file__))
# CPP_SRC_PATH = join(PROJ_DIR, 'cpp', 'src')
CPP_SRC_PATH = join('cpp', 'src')

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# gather up all the source files
# srcFiles = [join(PROJ_DIR, 'python', 'bolt', 'native.i')]
srcFiles = [join('python', 'bolt', 'native.i')]
includeDirs = [numpy_include]
paths = [CPP_SRC_PATH]
for path in paths:
    srcDir = path
    for root, dirNames, fileNames in os.walk(srcDir):
        for dirName in dirNames:
            absPath = os.path.join(root, dirName)
            if absPath.startswith("cpp/src/external"):
                continue
            print('adding dir to path: %s' % absPath)
            globStr = "%s/*.cpp" % absPath
            files = glob(globStr)
            if 'eigen/src' not in absPath:  # just include top level
                includeDirs.append(absPath)
            srcFiles += files

# set the compiler flags so it'll build on different platforms (feel free
# to file a  pull request with a fix if it doesn't work on yours)
# note that -march=native implies -mavx and -mavx2; Bolt requires AVX2
extra_args = ['-std=c++14',
              '-fno-rtti',
              '-march=native',
              '-ffast-math']
if sys.platform == 'darwin':
    extra_args.append('-mmacosx-version-min=10.9')
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9 -stdlib=libc++ -framework Accelerate'
    os.environ["CC"] = "g++"  # force compiling c as c++
else:  # based on Issue #4
    if "CC" not in os.environ:
        os.environ['CC'] = "clang"
    if "CXX" not in os.environ:
        os.environ['CXX'] = "clang++"
    # extra_args += ['-stdlib=libc++']
    # os.environ['CC'] = "clang"
    # os.environ['CXX'] = "clang++"
    # os.environ['LDFLAGS'] = '-lc++'
# else:
    # os.environ["CC"] = "clang++"  # force compiling c as c++


# inplace extension module
includeDirs += [join(PROJ_DIR, 'python', 'bolt')]

if 'EIGEN_INCLUDE_DIR' in os.environ:
    includeDirs += [
        os.environ['EIGEN_INCLUDE_DIR'],
        os.environ['EIGEN_INCLUDE_DIR'] + '/Eigen'
    ] 
else:
    includeDirs += [
        join(PROJ_DIR, 'cpp', 'src', 'external', 'eigen'),
        join(PROJ_DIR, 'cpp', 'src', 'external', 'eigen', 'Eigen')
    ]

nativeExt = Extension("bolt._bolt",  # must match cpp header name with leading _
                      srcFiles,
                      define_macros=[('NDEBUG', '1')],
                      include_dirs=includeDirs,
                      # swig_opts=['-c++', '-modern'],
                      swig_opts=['-c++'],
                      extra_compile_args=extra_args
                      # extra_link_args=['-stdlib=libc++'],
                      )

# ================================ Python modules

# glob_str = join('python', 'bolt') + '*.py'
# modules = [splitext(basename(path))[0] for path in glob(glob_str)]

# ================================ Call to setup()

# This ensures that the extension (which generates bolt.py) is built
# before py_modules are copied.

class CustomInstall(install):
    def run(self):
        self.run_command('build_ext')
        self.do_egg_install()

setup(
    cmdclass={'install': CustomInstall},
    name='pybolt',
    version='0.1.4',
    license='MPL',
    description='Fast approximate matrix and vector operations',
    author='Davis Blalock',
    author_email='dblalock@mit.edu',
    url='https://github.com/dblalock/bolt',
    download_url='https://github.com/dblalock/bolt/archive/0.1.tar.gz',
    packages=['bolt'],
    package_dir={'bolt': 'python/bolt'},
    py_modules=['python/bolt/bolt'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',
    ],
    keywords=[
        'Machine Learning', 'Compression', 'Big Data',
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    install_requires=[
        'numpy',
        'scikit-learn',
        #'kmc2'
        # 'sphinx_rtd_theme'  # for docs
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
