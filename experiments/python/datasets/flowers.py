#!/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np

from python import image_utils as imgs

from joblib import Memory
_memory = Memory('.', verbose=1)

DATADIR_101 = '../datasets/caltech/101_ObjectCategories'


def main():
    import matplotlib.pyplot as plt

    # caltech 101
    (X, y), label2cls = imgs.load_jpegs_from_dir(
        # TODO
        )

    if isinstance(X, np.ndarray):
        print("X shape: ", X.shape)
    else:
        print("X is a list of length", len(X))
        print("X[0] has shape: ", X[0].shape)
    print("y shape: ", y.shape)

    _, axes = plt.subplots(4, 4, figsize=(9, 9))

    for i, ax in enumerate(axes.ravel()):
        idx = np.random.choice(len(X))
        ax.imshow(X[idx])
        label = label2cls[y[idx]]
        ax.set_title(label)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
