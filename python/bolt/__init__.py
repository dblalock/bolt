#!/usr/bin/env python

import sklearn

# note that we import module generate py file, not the generated
# wrapper so (which is _bolt)
from bolt_api import *  # noqa

# from __future__ import print_function


def __bolt_debug_install():
    print "yep, bolt is installed"
