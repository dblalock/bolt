#!/usr/bin/env python

# from future import absolute_import, division, print_function

import os
import numpy as np
# import pathlib as pl
from sklearn import linear_model

from python.datasets import ampds, caltech, sharee
from python import window

from joblib import Memory
_memory = Memory('.', verbose=0)


_dir = os.path.dirname(os.path.abspath(__file__))
CIFAR10_DIR = os.path.join(_dir, '..', 'assets', 'cifar10-softmax')
CIFAR100_DIR = os.path.join(_dir, '..', 'assets', 'cifar100-softmax')


# ================================================================ types

class MatmulTask(object):

    def __init__(self, X_train, Y_train, X_test, Y_test, W_train, W_test=None,
                 name=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.W_train = W_train
        self.W_test = W_test if W_test is not None else W_train
        self.name = name

        self.train_mats = (self.X_train, self.Y_train, self.W_train)
        self.test_mats = (self.X_test, self.Y_test, self.W_test)

    def __str__(self):
        train_str = '{} @ {} = {}'.format(
            self.X_train.shape, self.W_train.shape, self.Y_train.shape)
        test_str = '{} @ {} = {}'.format(
            self.X_test.shape, self.W_test.shape, self.Y_test.shape)
        s = "train:\t{}\ntest:\t{}".format(train_str, test_str)
        if self.name:
            s = "---- {}\n{}".format(self.name, s)
        return s

    def validate(self, verbose=1, mse_thresh=1e-7):
        for (X, Y, W) in [self.train_mats, self.test_mats]:
            Y_hat = X @ W
            diffs = Y - Y_hat
            mse = np.mean(diffs * diffs) / np.var(Y)
            if verbose > 0:
                print("mse: ", mse)
            assert mse < mse_thresh


# class


# ================================================================ ampds

# TODO deterministic train test split

def load_ampd_data_mat():
    # return ampds.all_power_recordings()[0].data
    return ampds.all_weather_recordings()[0].data
    # print("mat.shape", mat.shape)


@_memory.cache
def load_ampd_x_y_w(window_len=16, verbose=1):
    data = load_ampd_data_mat()
    windows = window.sliding_window(
        data, ws=(window_len, data.shape[1]), ss=(1, 1))

    X = windows.reshape(windows.shape[0], -1)[:-1]
    Y = windows[1:, -1, :]  # targets are last row of next window

    N = len(X)
    N_train = N // 2
    # N_test = N - N_train
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    # fit the autoregressive model (just a filter)
    est = linear_model.LinearRegression(fit_intercept=False)
    # est.fit(X, Y)
    est.fit(X_train, Y_train)
    W = est.coef_.T

    if verbose > 0:
        print("ampd data.shape: ", data.shape)
        print("ampd windows.shape: ", windows.shape)
        print("ampd X shape: ", X.shape)
        print("ampd Y shape: ", Y.shape)

    # print(W.shape)
    # print(est.score(X, Y))
    # Y_hat = est.predict(X)
    # diffs = Y - Y_hat
    # print("normalized mse: ", np.mean(diffs * diffs) / np.var(Y))
    # Y_hat = X @ W
    # diffs = Y - Y_hat
    # print("normalized mse: ", np.mean(diffs * diffs) / np.var(Y))

    return X, Y, W  # Y_hat = X @ W


# def load_ampd_filter(windows):
#     est = linear_model.LinearRegression()
#     X = windows[:-1]
#     Y = windows[]
#     est.fit(windows)
#     pass

# ================================================================ ECG (sharee)

def load_x_y_w_for_ar_model(data, window_len=8, verbose=1, N_train=-1):
    windows = window.sliding_window(
        data, ws=(window_len, data.shape[1]), ss=(1, 1))

    X = windows.reshape(windows.shape[0], -1)[:-1]
    Y = windows[1:, -1, :]  # targets are last row of next window

    N = len(X)
    if N_train < data.shape[1]:
        N_train = N // 2  # TODO more flexible train/test split
    # N_test = N - N_train
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    # fit the autoregressive model (just a filter)
    est = linear_model.LinearRegression(fit_intercept=False)
    # est.fit(X, Y)
    est.fit(X_train, Y_train)
    W = est.coef_.T

    if verbose > 0:
        print("ts ar model: data.shape: ", data.shape)
        print("ts ar model: windows.shape: ", windows.shape)
        print("ts ar model: X shape: ", X.shape)
        print("ts ar model: Y shape: ", Y.shape)

    # print(W.shape)
    # print(est.score(X, Y))
    # Y_hat = est.predict(X)
    # diffs = Y - Y_hat
    # print("normalized mse: ", np.mean(diffs * diffs) / np.var(Y))
    # Y_hat = X @ W
    # diffs = Y - Y_hat
    # print("normalized mse: ", np.mean(diffs * diffs) / np.var(Y))

    # return X_test, Y_test, W  # Y_hat = X @ W
    return MatmulTask(X_train=X_train, Y_train=Y_train,
                      X_test=X_test, Y_test=Y_test, W_train=W)


def load_ecg_x_y_w_for_recording(recording, window_len=8):
    return load_x_y_w_for_ar_model(recording, window_len=window_len)


def load_ecg_recordings():
    # return list(sharee.load_recordings())
    return sharee.load_recordings()  # generator, since takes lots of memory


def load_ecg_tasks(window_len=8):
    recordings = load_ecg_recordings()
    for recording in recordings:
        yield load_ecg_x_y_w_for_recording(recording, window_len=window_len)


# ================================================================ caltech

# TODO deterministic train test split

def load_caltech_imgs(ntrain_classes=50, limit_per_class=10):
    # imgs = caltech.load_caltech101(resample=None, crop=None)
    (imgs, y), label2name = caltech.load_caltech101(
        resample=None, crop=None, limit_per_class=limit_per_class)
    # print(len(imgs))
    # print(sum([img.size for img in imgs]))
    # print("class counts: ", np.bincount(y))

    # split by class; more relaxed than assuming you have examples from
    # the same dataset/class you're later going to apply your filter to
    train_idxs = np.where(y < ntrain_classes)[0]
    test_idxs = np.where(y >= ntrain_classes)[0]
    imgs_train = [imgs[i] for i in train_idxs]
    imgs_test = [imgs[i] for i in test_idxs]

    return imgs_train, imgs_test


def load_caltech_img_ids(ntrain_classes=50, limit_per_class=10):
    (imgs, y), label2name = caltech.load_caltech101_ids(
        limit_per_class=limit_per_class)

    # split by class; more relaxed than assuming you have examples from
    # the same dataset/class you're later going to apply your filter to
    train_idxs = np.where(y < ntrain_classes)[0]
    test_idxs = np.where(y >= ntrain_classes)[0]
    imgs_ids_train = [imgs[i] for i in train_idxs]
    imgs_ids_test = [imgs[i] for i in test_idxs]

    return imgs_ids_train, imgs_ids_test


def load_dummy_caltech_filter_3x3(order='hwc'):
    filt = [[-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1]]
    if order == 'hwc':
        filt = np.array(filt)[..., np.newaxis]
        filt = np.tile(filt, (1, 1, 3))
    else:
        assert order == 'chw'
        filt = np.array(filt)[np.newaxis, ...]
        filt = np.tile(filt, (3, 1, 1))
    # print(filt)
    return filt


def _lift_grayscale_filt_to_rgb(filt, order='hwc'):
    if order == 'hwc':
        filt = np.array(filt)[..., np.newaxis]
        filt = np.tile(filt, (1, 1, 3))
    else:
        assert order == 'chw'
        filt = np.array(filt)[np.newaxis, ...]
        filt = np.tile(filt, (3, 1, 1))
    return filt


def _lift_vert_filter_to_rgb_pair(filt_v, order='hwc'):
    filt_v = np.array(filt_v)
    filt_h = np.ascontiguousarray(filt_v.T)
    filt_v = _lift_grayscale_filt_to_rgb(filt_v)
    filt_h = _lift_grayscale_filt_to_rgb(filt_h)

    return filt_v, filt_h


def load_filters_sobel_3x3(order='hwc'):
    filt_v = [[-1,  0, 1],
              [-2, 0, 2],
              [-1,  0, 1]]
    filt_v = np.array(filt_v, dtype=np.float32) / 2.
    return _lift_vert_filter_to_rgb_pair(filt_v)


def load_filters_sobel_5x5(order='hwc'):
    filt_v = [[-5,  -4, 0, 4,  5],
              [-8, -10, 0, 10, 8],
              [-10, 20, 0, 20, 10],
              [-8, -10, 0, 10, 8],
              [-5,  -4, 0, 4,  5]]
    filt_v = np.array(filt_v, dtype=np.float32) / 20.
    return _lift_vert_filter_to_rgb_pair(filt_v)


def _filters_list_to_mat(filters):
    filters_flat = [filt.ravel() for filt in filters]
    return np.vstack(filters_flat).T


def caltech_x_y_for_img(img, filt_spatial_shape, filters_list=None, W=None):
    # W = filt.ravel()
    # img = img[np.newaxis, ...]
    # windows = window.extract_conv2d_windows(img, filt_shape=filt.shape[1:])

    # print(windows.shape)

    # extract and flatten windows into rows of X matrix
    windows = window.extract_conv2d_windows(img, filt_shape=filt_spatial_shape)
    # filt_sz = filters_list[0].size
    filt_sz = img.shape[-1] * np.prod(filt_spatial_shape)
    X = windows.reshape(-1, filt_sz)

    # compute each column of y matrix
    W = _filters_list_to_mat(filters_list) if W is None else W
    # filters need to all be the same shape
    # shapes = [filt.shape for filt in filters_list]
    # assert all([shape == shapes[0] for shape in shapes])

    # Y = np.zeros((X.shape[0], len(filters_list)), dtype=np.float32)
    # for i, filt in enumerate(filters_list):
    #     broadcast_filt = filt.reshape(1, 1, *filt.shape)
    #     dot_prods = np.sum(windows * broadcast_filt, axis=(2, 3, 4))
    #     Y[:, i] = dot_prods.reshape(len(X))

    return X, X @ W


# def _load_caltech_train(filters, filt_spatial_shape):
def _load_caltech_train(W, filt_spatial_shape):
    img_ids_train, img_ids_test = load_caltech_img_ids()
    train_imgs = [caltech.load_caltech_img(img_id) for img_id in img_ids_train]

    #
    # uncomment to plot imgs to make sure this is working
    #
    # which_idxs = np.random.randint(len(train_imgs), size=16)
    # # imgs = [train_imgs[i] for i in which_idxs]
    # import matplotlib.pyplot as plt
    # _, axes = plt.subplots(4, 4, figsize=(9, 9))
    # for i, idx in enumerate(which_idxs):
    #     axes.ravel()[i].imshow(train_imgs[idx])
    # plt.show()

    train_mats = [caltech_x_y_for_img(img,
                  filt_spatial_shape=filt_spatial_shape, W=W)
                  for img in train_imgs]
    Xs, Ys = list(zip(*train_mats))
    X_train = np.vstack(Xs)
    Y_train = np.vstack(Ys)

    return X_train, Y_train


@_memory.cache  # cache raw images to avoid IO, but dynamically produce windows
def _load_caltech_test_imgs():
    _, test_ids = load_caltech_img_ids()
    return [caltech.load_caltech_img(img_id) for img_id in test_ids]


def load_caltech_tasks():
    # imgs_train, imgs_test = load_caltech_imgs()
    filters = load_filters_sobel_3x3()
    filt_spatial_shape = (3, 3)
    W = _filters_list_to_mat(filters)

    X_train, Y_train = _load_caltech_train(
        W=W, filt_spatial_shape=filt_spatial_shape)
    test_imgs = _load_caltech_test_imgs()

    print("X train shape: ", X_train.shape)
    print("X train nbytes: ", X_train.nbytes)
    print("Y train shape: ", Y_train.shape)
    print("Y train nbytes: ", Y_train.nbytes)

    print("size of test imgs (not windows): ",
          sum([img.nbytes for img in test_imgs]))

    for img in test_imgs:
        X_test, Y_test = caltech_x_y_for_img(
            img, filt_spatial_shape=filt_spatial_shape, W=W)
        yield MatmulTask(X_train=X_train, Y_train=Y_train, W_train=W,
                         X_test=X_test, Y_test=Y_test, W_test=W)


def test_caltech_loading():
    imgs_train, imgs_test = load_caltech_imgs()
    filt = load_dummy_caltech_filter_3x3()
    imgs = imgs_train

    print("imgs[0].shape", imgs[0].shape)
    print("filt shape: ", filt.shape)
    X, Y = caltech_x_y_for_img(imgs[0], filt)
    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)

    # yep, looks like these are equivalent
    flat_filt = filt.ravel()
    Y_hat = X @ flat_filt
    diffs = Y - Y_hat
    mse = np.sum(diffs * diffs) / np.var(Y)
    print("mse: ", mse)
    assert mse < 1e-10


# ================================================================ cifar

def load_cifar10_task():
    SOFTMAX_INPUTS_TRAIN_PATH = 'cifar10_softmax_inputs_train.npy'
    SOFTMAX_OUTPUTS_TRAIN_PATH = 'cifar10_softmax_outputs_train.npy'
    SOFTMAX_INPUTS_TEST_PATH = 'cifar10_softmax_inputs_test.npy'
    SOFTMAX_OUTPUTS_TEST_PATH = 'cifar10_softmax_outputs_test.npy'
    SOFTMAX_W_PATH = 'cifar10_softmax_W.npy'
    SOFTMAX_B_PATH = 'cifar10_softmax_b.npy'

    def load_mat(fname):
        fpath = os.path.join(CIFAR10_DIR, fname)
        return np.load(fpath)

    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)

    # we aren't going to store or approximate the biases, so just subtract
    # off their contributions at the start
    Y_train -= b
    Y_test -= b

    return MatmulTask(X_train, Y_train, X_test, Y_test, W,
                      name='CIFAR-10 Softmax')


def load_cifar100_task():
    SOFTMAX_INPUTS_TRAIN_PATH = 'cifar100_softmax_inputs_train.npy'
    SOFTMAX_OUTPUTS_TRAIN_PATH = 'cifar100_softmax_outputs_train.npy'
    SOFTMAX_INPUTS_TEST_PATH = 'cifar100_softmax_inputs_test.npy'
    SOFTMAX_OUTPUTS_TEST_PATH = 'cifar100_softmax_outputs_test.npy'
    SOFTMAX_W_PATH = 'cifar100_softmax_W.npy'
    SOFTMAX_B_PATH = 'cifar100_softmax_b.npy'

    def load_mat(fname):
        fpath = os.path.join(CIFAR100_DIR, fname)
        return np.load(fpath)

    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)

    # we aren't going to store or approximate the biases, so just subtract
    # off their contributions at the start
    Y_train -= b
    Y_test -= b

    return MatmulTask(X_train, Y_train, X_test, Y_test, W,
                      name='CIFAR-100 Softmax')


# ================================================================ main

def main():
    load_caltech_tasks()

    # load_ampd_data_mat()
    # load_ampd_windows()
    # load_ampd_x_y_w()
    # load_caltech_imgs()
    # test_caltech_loading()

    # task = load_cifar10_task()
    # print(task)
    # task.validate()
    # task = load_cifar100_task()
    # print(task)
    # task.validate()

    # load_dummy_caltech_filter_3x3()


if __name__ == '__main__':
    main()
