#!/bin/env python

from __future__ import absolute_import, division, print_function

from scipy import io
import numpy as np
import os

from joblib import Memory
_memory = Memory('.', verbose=1)


DATADIR = '../datasets/svhn'
TRAIN_PATH = os.path.join(DATADIR, 'train_32x32.mat')
TEST_PATH = os.path.join(DATADIR, 'test_32x32.mat')
EXTRA_PATH = os.path.join(DATADIR, 'extra_32x32.mat')


def extract_data_from_mat_file(path):
    matlab_dict = io.loadmat(path)
    X, y = matlab_dict['X'], matlab_dict['y'].ravel()
    X = np.transpose(X, (3, 0, 1, 2))

    # make classes be 0-9 instead of 1-10; this way the classes line up
    # with the actual digits
    y[y == 10] = 0

    assert len(y.shape) == 1
    assert X.shape[0] == len(y)
    assert X.shape[1] == 32
    assert X.shape[2] == 32
    assert X.shape[-1] == 3

    return X, y


@_memory.cache
def load_data():
    X_train, y_train = extract_data_from_mat_file(TRAIN_PATH)
    X_test, y_test = extract_data_from_mat_file(TEST_PATH)

    return (X_train, y_train), (X_test, y_test)


def load_extra_data():
    return extract_data_from_mat_file(EXTRA_PATH)


def main():
    import matplotlib.pyplot as plt

    (X_train, y_train), (X_test, y_test) = load_data()

    # hacky way to visualize extra data using same code
    # X_extra, y_extra = load_extra_data()
    # X_train, X_test = X_extra, X_extra
    # y_train, y_test = y_extra, y_extra

    _, axes = plt.subplots(4, 4, figsize=(9, 9))

    for i, ax in enumerate(axes.ravel()):
        X = X_test if i % 2 else X_train
        y = y_test if i % 2 else y_train

        idx = np.random.choice(X.shape[0])
        ax.imshow(X[idx])
        ax.set_title("class = {}".format(y[idx]))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
