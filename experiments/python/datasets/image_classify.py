#!/bin/env python

from __future__ import print_function

import numpy as np
import os
import warnings
import h5py
from sklearn.datasets import load_digits
import keras
from keras.preprocessing import image

# from python import imagenet, svhn, caltech
# from python.datasets import caltech
from . import imagenet
from . import svhn
from .data_utils import stratified_split_train_test

from joblib import Memory
_memory = Memory('.', verbose=1)

# DATA_DIR = os.path.expanduser('~/Desktop/datasets/nn-search')
DATA_DIR = os.path.expanduser('data')
join = os.path.join


DEFAULT_AUG_KWARGS = {
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True
}


class LabeledDataset(object):
    __slots__ = 'name X_train y_train X_test y_test _collection'.split()

    def __init__(self, name, X_train, y_train, X_test=None, y_test=None):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def generators(self, batch_size, augment=True,
                   preprocessing_function=None, aug_kwargs=None):
        _aug_kwargs = DEFAULT_AUG_KWARGS
        if aug_kwargs is not None:
            _aug_kwargs.update(aug_kwargs)
        if not augment:
            _aug_kwargs = {}

        nclasses = len(np.unique(self.y_train))
        y_train = keras.utils.to_categorical(self.y_train, num_classes=nclasses)
        y_test = keras.utils.to_categorical(self.y_test, num_classes=nclasses)

        train_datagen = image.ImageDataGenerator(
            preprocessing_function=preprocessing_function, **_aug_kwargs)
        train_generator = train_datagen.flow(
            self.X_train, y_train, batch_size=batch_size)

        test_datagen = image.ImageDataGenerator(
            preprocessing_function=preprocessing_function)
        test_generator = test_datagen.flow(
            self.X_test, y_test, batch_size=batch_size)

        return train_generator, test_generator


class HugeLabeledDataset(object):

    def __init__(self, name, train_dir, test_dir,
                 train_nsamples=None, test_nsamples=None):
        self.name = name
        # self.train_dir = os.path.abspath(train_dir)
        # self.test_dir = os.path.abspath(test_dir)
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_nsamples = int(train_nsamples or -1)
        self.test_nsamples = int(test_nsamples or -1)

    def generators(self, batch_size=None, augment=True,
                   preprocessing_function=None, aug_kwargs=None,
                   train_batch_size=None, test_batch_size=None,
                   **flow_kwargs):
        _aug_kwargs = DEFAULT_AUG_KWARGS
        if aug_kwargs is not None:
            _aug_kwargs.update(aug_kwargs)
        if not augment:
            _aug_kwargs = {}

        flow_kwargs = flow_kwargs or {}
        flow_kwargs.setdefault('target_size', (224, 224))
        flow_kwargs.setdefault('class_mode', 'categorical')

        train_generator = None
        test_generator = None

        if self.train_dir:
            train_batch_size = int(train_batch_size or batch_size)
            flow_kwargs['batch_size'] = train_batch_size
            print("HugeLabeledDataset: creating flow from train dir: ",
                  self.train_dir)
            train_datagen = image.ImageDataGenerator(
                preprocessing_function=preprocessing_function, **_aug_kwargs)
            train_generator = train_datagen.flow_from_directory(
                self.train_dir, **flow_kwargs)

        if self.test_dir:
            test_batch_size = int(test_batch_size or batch_size)
            flow_kwargs['batch_size'] = test_batch_size
            print("HugeLabeledDataset: creating flow from test dir: ",
                  self.test_dir)
            test_datagen = image.ImageDataGenerator(
                preprocessing_function=preprocessing_function)
            test_generator = test_datagen.flow_from_directory(
                self.test_dir, **flow_kwargs)

        return train_generator, test_generator


class Random:
    UNIFORM = 'uniform'
    GAUSS = 'gauss'
    WALK = 'walk'
    BLOBS = 'blobs'


DIGITS = 'Digits'
MNIST = 'MNIST'
FASHION_MNIST = 'FashionMNIST'
CIFAR10 = 'Cifar10'
CIFAR100 = 'Cifar100'
SVHN = 'SVHN'
CALTECH101 = 'Caltech101'
CALTECH256 = 'Caltech256'
CUB200 = 'CUB200'
FLOWERS102 = 'Flowers102'
INDOOR67 = 'Indoor67'
IMAGENET_TINY = 'TinyImagenet'  # 64x64, 200? classes
IMAGENET_10_CLASSES = 'ImageNet-10-Classes'      # full res, 10cls, 1k/cls
IMAGENET_100_CLASSES = 'ImageNet-100-Classes'    # full res, 100cls, 1k/cls
IMAGENET_1_EXAMPLE = 'ImageNet-1-Example'        # full res, 1k cls, 1/cls
IMAGENET_10_EXAMPLES = 'ImageNet-10-Examples'    # full res, 1k cls, 10/cls
IMAGENET_25_EXAMPLES = 'ImageNet-25-Examples'    # full res, 1k cls, 25/cls
IMAGENET_50_EXAMPLES = 'ImageNet-50-Examples'    # full res, 1k cls, 50/cls
IMAGENET_100_EXAMPLES = 'ImageNet-100-Examples'  # full res, 1k cls, 100/cls
IMAGENET_64PX = 'ImageNet64'                     # 64x64, all examples
IMAGENET = 'ImageNet'
IMAGENET_ONE_OF_EACH = 'ImagenetOneOfEach'
MINIPLACES = 'Miniplaces'

ALL_IMAGENET_DATASETS = [
    IMAGENET, IMAGENET_64PX, IMAGENET_TINY, IMAGENET_ONE_OF_EACH,
    IMAGENET_10_CLASSES, IMAGENET_100_CLASSES,
    IMAGENET_1_EXAMPLE, IMAGENET_10_EXAMPLES, IMAGENET_100_EXAMPLES]

ALL_KERAS_DATASETS = [MNIST, CIFAR10, CIFAR100, FASHION_MNIST]


def _load_file(fname, *args, **kwargs):
    fname = os.path.join(DATA_DIR, fname)
    print("trying to load file at path: {}".format(fname))
    if fname.split('.')[-1] == 'txt':
        return np.loadtxt(fname, *args, **kwargs)
    return np.load(fname, *args, **kwargs)


def _load_digits_X_y(ntrain=1000):
    X, y = load_digits(return_X_y=True)
    X_train, X_test = X[:ntrain], X[ntrain:]
    y_train, y_test = y[:ntrain], y[ntrain:]
    return LabeledDataset('Digits', X_train, y_train, X_test, y_test)
    # return X[:-nqueries], X[-nqueries:]  # X, Q


def _load_keras_dset(which_dataset):
    from keras import datasets as kd
    dataClass = {CIFAR10: kd.cifar10,
                 CIFAR100: kd.cifar100,
                 MNIST: kd.mnist,
                 FASHION_MNIST: kd.fashion_mnist}[which_dataset]
    (X_train, y_train), (X_test, y_test) = dataClass.load_data()
    pretty_name = str(which_dataset).split('.')[-1].split("'")[0]
    return LabeledDataset(pretty_name, X_train, y_train, X_test, y_test)


def load_imagenet_64(limit_ntrain=-1):
    # if we're not going to use the whole training set, don't even load in all
    # the files it's split into (necessary unless you have >18GB of free RAM)
    which_file_idxs = None
    if limit_ntrain > 0:
        nchunks = int(np.ceil(
            limit_ntrain / imagenet.IMAGENET_64_TRAIN_CHUNK_NSAMPLES))
        which_file_idxs = np.arange(1, nchunks + 1)
    X_train, y_train = imagenet.load_train_data_64x64(
        which_file_idxs=which_file_idxs)
    X_test, y_test = imagenet.load_test_data_64x64()
    return LabeledDataset(IMAGENET_64PX, X_train, y_train, X_test, y_test,
                          train_nsamples=1e6)


def load_imagenet_tiny():
    X_train, y_train = imagenet.load_train_data_tiny()
    X_test, y_test = imagenet.load_test_data_tiny()
    return LabeledDataset(IMAGENET_TINY, X_train, y_train, X_test, y_test)


def load_imagenet_one_of_each():
    X, y = imagenet.load_data_one_of_each()
    return LabeledDataset(IMAGENET_ONE_OF_EACH, X, y, X, y, train_nsamples=1e3)


def load_imagenet(load_train=True, load_val=True):
    train_path = imagenet.IMAGENET_TRAIN_PATH if load_train else None
    test_path = imagenet.IMAGENET_TEST_PATH if load_val else None
    return HugeLabeledDataset(
        IMAGENET, train_path, test_path,
        train_nsamples=1281167, test_nsamples=50e3)


def load_imagenet_10_classes(load_train=True, load_val=True):
    train_path = imagenet.IMAGENET_10_CLASSES_TRAIN_PATH if load_train else None
    test_path = imagenet.IMAGENET_10_CLASSES_TEST_PATH if load_val else None
    return HugeLabeledDataset(
        IMAGENET_10_CLASSES, train_path, test_path,
        train_nsamples=13000, test_nsamples=500)


def load_imagenet_100_classes(load_train=True, load_val=True):
    train_path = imagenet.IMAGENET_100_CLASSES_TRAIN_PATH \
        if load_train else None
    test_path = imagenet.IMAGENET_100_CLASSES_TEST_PATH if load_val else None
    return HugeLabeledDataset(IMAGENET_100_CLASSES, train_path, test_path,
                              train_nsamples=129395, test_nsamples=5000)


def load_imagenet_1_example(load_train=True, load_val=True):
    train_path = imagenet.IMAGENET_1_EXAMPLE_TRAIN_PATH \
        if load_train else None
    test_path = imagenet.IMAGENET_TEST_PATH if load_val else None
    return HugeLabeledDataset(IMAGENET_10_EXAMPLES, train_path, test_path,
                              train_nsamples=1e3, test_nsamples=50e3)


def load_imagenet_10_examples(load_train=True, load_val=True):
    train_path = imagenet.IMAGENET_10_EXAMPLES_TRAIN_PATH \
        if load_train else None
    test_path = imagenet.IMAGENET_TEST_PATH if load_val else None
    return HugeLabeledDataset(IMAGENET_10_EXAMPLES, train_path, test_path,
                              train_nsamples=10e3, test_nsamples=50e3)


def load_imagenet_25_examples(load_train=True, load_val=True):
    train_path = imagenet.IMAGENET_25_EXAMPLES_TRAIN_PATH \
        if load_train else None
    test_path = imagenet.IMAGENET_TEST_PATH if load_val else None
    return HugeLabeledDataset(IMAGENET_25_EXAMPLES, train_path, test_path,
                              train_nsamples=25e3, test_nsamples=50e3)


def load_imagenet_50_examples(load_train=True, load_val=True):
    train_path = imagenet.IMAGENET_50_EXAMPLES_TRAIN_PATH \
        if load_train else None
    test_path = imagenet.IMAGENET_TEST_PATH if load_val else None
    return HugeLabeledDataset(IMAGENET_50_EXAMPLES, train_path, test_path,
                              train_nsamples=50e3, test_nsamples=50e3)


def load_imagenet_100_examples(load_train=True, load_val=True):
    train_path = imagenet.IMAGENET_100_EXAMPLES_TRAIN_PATH \
        if load_train else None
    test_path = imagenet.IMAGENET_TEST_PATH if load_val else None
    return HugeLabeledDataset(IMAGENET_10_EXAMPLES, train_path, test_path,
                              train_nsamples=100e3, test_nsamples=50e3)


def _load_miniplaces():
    path = '/data/ddmg/neuro/datasets/Miniplaces/miniplaces.h5'
    with h5py.File(path, 'r') as hf:
        X_train = hf['X_train'][()]
        Y_train = hf['Y_train'][()]
        X_val = hf['X_val'][()]
        Y_val = hf['Y_val'][()]

    return LabeledDataset(MINIPLACES, X_train, Y_train, X_val, Y_val)


def _load_svhn():
    (X_train, y_train), (X_test, y_test) = svhn.load_data()
    return LabeledDataset(SVHN, X_train, y_train, X_test, y_test)


def load_caltech101():
    data_dir = '../datasets/caltech/101_ObjectCategories'
    return HugeLabeledDataset(CALTECH101, data_dir, None)
    # (X, y), _ = caltech.load_caltech101()
    # return LabeledDataset(IMAGENET_ONE_OF_EACH, X, y, X, y)


def load_caltech256():
    data_dir = '../datasets/caltech/256_ObjectCategories'
    return HugeLabeledDataset(CALTECH256, data_dir, None)
    # (X, y), _ = caltech.load_caltech256()
    # return LabeledDataset(IMAGENET_ONE_OF_EACH, X, y, X, y)


def load_flowers102():
    data_dir = '../datasets/flowers102'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    return HugeLabeledDataset(FLOWERS102, train_dir, test_dir,
                              train_nsamples=1020, test_nsamples=6149)


def load_cub200():  # note that this is 2011 version of CUB200
    data_dir = '../datasets/cub200'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    return HugeLabeledDataset(CUB200, train_dir, test_dir,
                              train_nsamples=5994, test_nsamples=5794)


def load_indoor67():  # this is the subset with predefined train vs test split
    data_dir = '../datasets/indoor67'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    return HugeLabeledDataset(INDOOR67, train_dir, test_dir,
                              train_nsamples=(67 * 80), test_nsamples=(67 * 20))


# @_memory.cache
def load_dataset(which_dataset, norm_mean=False, norm_len=False,
                 flatten=False, Ntrain=-1, Ntest=-1, ensure_channels=False,
                 test_frac=None, scale_to_0_1=False):
    if which_dataset == DIGITS:
        dset = _load_digits_X_y()
    elif which_dataset in ALL_KERAS_DATASETS:
        dset = _load_keras_dset(which_dataset)
    elif which_dataset == IMAGENET_64PX:
        dset = load_imagenet_64(limit_ntrain=Ntrain)
    elif which_dataset == IMAGENET_TINY:
        dset = load_imagenet_tiny()
    elif which_dataset == IMAGENET_ONE_OF_EACH:
        dset = load_imagenet_one_of_each()
    elif which_dataset == MINIPLACES:
        dset = _load_miniplaces()
    elif which_dataset == SVHN:
        dset = _load_svhn()
    elif which_dataset == CALTECH101:
        return load_caltech101()
    elif which_dataset == CALTECH256:
        return load_caltech256()
    elif which_dataset == CUB200:
        return load_cub200()
    elif which_dataset == FLOWERS102:
        return load_flowers102()
    elif which_dataset == IMAGENET:
        return load_imagenet()
    elif which_dataset == IMAGENET_10_CLASSES:
        return load_imagenet_10_classes()
    elif which_dataset == IMAGENET_100_CLASSES:
        return load_imagenet_100_classes()
    elif which_dataset == IMAGENET_1_EXAMPLE:
        return load_imagenet_1_example()
    elif which_dataset == IMAGENET_10_EXAMPLES:
        return load_imagenet_10_examples()
    elif which_dataset == IMAGENET_25_EXAMPLES:
        return load_imagenet_25_examples()
    elif which_dataset == IMAGENET_50_EXAMPLES:
        return load_imagenet_50_examples()
    elif which_dataset == IMAGENET_100_EXAMPLES:
        return load_imagenet_100_examples()
    else:
        raise ValueError("unrecognized dataset {}".format(which_dataset))

    if isinstance(dset, HugeLabeledDataset):
        # only has flow_from_directory() generators; no postprocessing
        # possible, so go ahead and return immediately
        return dset

    train_is_test = (dset.X_train.base is dset.X_test) or \
        (dset.X_test.base is dset.X_train)
    train_test_equal = np.array_equal(dset.X_train[:10], dset.X_test[:10])
    train_test_same = train_is_test or train_test_equal

    if train_test_same:
        if test_frac is None:
            warnings.warn("WARNING: Training data is also the test data! "
                          "Reversing order of test data. Consider passing "
                          "test_frac > 0 to automatically perform a "
                          "stratified train-test split.")
            dset.X_test = dset.X_test[::-1]
        else:
            X_train, X_test, y_train, y_test = stratified_split_train_test(
                dset.X_train, dset.y_train, train_frac=(1. - test_frac))
            dset = LabeledDataset(dset.name, X_train, y_train, X_test, y_test)

            train_is_test = False
            train_test_equal = False
            train_test_same = False
    if train_is_test:
        dset.X_test = np.copy(dset.X_test)
        dset.y_test = np.copy(dset.y_test)
        train_is_test = False

    if flatten:
        dset.X_train = dset.X_train.reshape(dset.X_train.shape[0], -1)
        dset.X_test = dset.X_test.reshape(dset.X_test.shape[0], -1)

    dset.X_train = dset.X_train.astype(np.float32)
    dset.X_test = dset.X_test.astype(np.float32)
    X_train = dset.X_train
    X_test = dset.X_test

    if Ntrain > 0:
        dset.X_train = X_train[:Ntrain]
        dset.y_train = dset.y_train[:Ntrain]
    if Ntest > 0:
        dset.X_test = np.copy(X_test[:Ntest])
        dset.y_test = np.copy(dset.y_test[:Ntest])

    if scale_to_0_1:
        min_X = min(np.min(dset.X_train), np.min(dset.X_test))
        max_X = max(np.max(dset.X_train), np.max(dset.X_test))
        dset.X_train = (dset.X_train - min_X) / max_X
        # if not train_is_test:
        dset.X_test = (dset.X_test - min_X) / max_X
    if norm_mean:
        means = np.mean(dset.X_train, axis=0)
        dset.X_train -= means
        # if not train_is_test:  # don't subtract means twice from same array
        dset.X_test -= means
    if norm_len:
        dset.X_train /= np.linalg.norm(dset.X_train, axis=1, keepdims=True)
        # if not train_is_test:  # don't divide by norms twice on same array
        dset.X_test /= np.linalg.norm(dset.X_test, axis=1, keepdims=True)

    if ensure_channels:
        import keras.backend as K  # don't import keras unless we need it
        if len(X_train.shape) == 3:  # no channels; e.g., MNIST
            img_rows, img_cols = X_train.shape[-2], X_train.shape[-1]
            # K.set_image_data_format('channels_last')  # for argmax layer
            if K.image_data_format() == 'channels_first':
                dset.X_train = dset.X_train.reshape(
                    X_train.shape[0], 1, img_rows, img_cols)
                dset.X_test = dset.X_test.reshape(
                    X_test.shape[0], 1, img_rows, img_cols)
            else:
                dset.X_train = dset.X_train.reshape(
                    X_train.shape[0], img_rows, img_cols, 1)
                dset.X_test = dset.X_test.reshape(
                    X_test.shape[0], img_rows, img_cols, 1)

    return dset

    # if D_multiple_of > 1:
    #     X_train = ensure_num_cols_multiple_of(X_train, D_multiple_of)
    #     X_test = ensure_num_cols_multiple_of(X_test, D_multiple_of)
    #     Q = ensure_num_cols_multiple_of(Q, D_multiple_of)

    # return X_train, Q, X_test, true_nn
