#!/bin/env/python

"""utility functions for data munging"""

from __future__ import absolute_import, division, print_function

import numpy as np
import sklearn


def split_train_test(X, Y, train_frac=.8, random_state=123):
    """Returns X_train, X_test, y_train, y_test"""
    np.random.seed(123)
    return sklearn.model_selection.train_test_split(
        X, Y, train_size=train_frac, random_state=random_state)


def stratified_split_train_test(X, Y, train_frac=.8, random_state=123):
    """Returns X_train, X_test, y_train, y_test"""
    return sklearn.model_selection.train_test_split(
        X, Y, train_size=train_frac, stratify=Y, random_state=random_state)
