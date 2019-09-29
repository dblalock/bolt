#!/bin/env python

# from __future__ import absolute_import, division, print_function
from __future__ import division, print_function

import numpy as np

from . import paths
from . import image_utils as imgs

from joblib import Memory
_memory = Memory('.', verbose=1)


DATADIR_101 = paths.CALTECH_101
DATADIR_256 = paths.CALTECH_256


# _DEFAULT_CALTECH_KWARGS = dict(resample=(224, 224), crop='center', verbose=2)
_DEFAULT_CALTECH_KWARGS = dict(resample=(224, 224), crop='center')
_CALTECH_101_KWARGS = dict(
    dirpath=DATADIR_101, remove_classes='BACKGROUND_Google')
_CALTECH_256_KWARGS = dict(
    dirpath=DATADIR_256, remove_classes='257.clutter')


@_memory.cache
def load_caltech101(**kwargs):
    [kwargs.setdefault(*item) for item in _DEFAULT_CALTECH_KWARGS.items()]
    return imgs.load_jpegs_from_dir(**_CALTECH_101_KWARGS, **kwargs)


@_memory.cache
def load_caltech256(**kwargs):
    [kwargs.setdefault(*item) for item in _DEFAULT_CALTECH_KWARGS.items()]
    return imgs.load_jpegs_from_dir(**_CALTECH_256_KWARGS, **kwargs)


@_memory.cache
def load_caltech101_ids(**kwargs):
    return imgs.load_jpegs_from_dir(
        **_CALTECH_101_KWARGS, only_return_path=True, **kwargs)


@_memory.cache
def load_caltech256_ids(**kwargs):
    return imgs.load_jpegs_from_dir(
        **_CALTECH_256_KWARGS, only_return_path=True, **kwargs)


# @_memory.cache
def load_caltech_img(img_id, **kwargs):
    [kwargs.setdefault(*item) for item in _DEFAULT_CALTECH_KWARGS.items()]
    path = img_id  # load_jpegs_from_dir returns abs path as id
    return imgs.load_jpg(path, **kwargs).astype(np.float32)
    # img = imgs.load_jpg(path, **kwargs).astype(np.float32)
    # print("img.shape", img.shape)
    # assert img.shape[:2] == (224, 224)
    # return img


def main():
    import matplotlib.pyplot as plt

    # caltech 101
    (X, y), label2cls = imgs.load_jpegs_from_dir(
        # DATADIR_101, remove_classes='BACKGROUND_Google')
        # DATADIR_101, remove_classes='BACKGROUND_Google', crop='center')
        DATADIR_101, remove_classes='BACKGROUND_Google', pad='square')
    #     # DATADIR_101, remove_classes='BACKGROUND_Google', resample=(224, 224))

    # caltech 256
    # (X, y), label2cls = imgs.load_jpegs_from_dir(
    #     DATADIR_256, remove_classes='257.clutter', verbose=2)

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
