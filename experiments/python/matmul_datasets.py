#!/usr/bin/env python

# from future import absolute_import, division, print_function

import os
import numpy as np
# import pathlib as pl
from sklearn import linear_model
from scipy import signal

from python.datasets import caltech, sharee, incart, ucr
from python import misc_algorithms as algo
from python import window

from joblib import Memory
# _memory = Memory('.', verbose=0, compress=7)  # compression between 1 and 9
# _memory = Memory('.', verbose=0, compress=3)  # compression between 1 and 9
_memory = Memory('.', verbose=0)


_dir = os.path.dirname(os.path.abspath(__file__))
CIFAR10_DIR = os.path.join(_dir, '..', 'assets', 'cifar10-softmax')
CIFAR100_DIR = os.path.join(_dir, '..', 'assets', 'cifar100-softmax')


# ================================================================ types

class MatmulTask(object):

    def __init__(self, X_train, Y_train, X_test, Y_test, W_train, W_test=None,
                 name=None, info=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.W_train = W_train
        self.W_test = W_test if W_test is not None else W_train
        self.name = name
        self.info = info if info is not None else {}

        self.train_mats = (self.X_train, self.Y_train, self.W_train)
        self.test_mats = (self.X_test, self.Y_test, self.W_test)
        self.initial_hashes = self._hashes()

    def __str__(self):
        train_str = '{} @ {} = {}'.format(
            self.X_train.shape, self.W_train.shape, self.Y_train.shape)
        test_str = '{} @ {} = {}'.format(
            self.X_test.shape, self.W_test.shape, self.Y_test.shape)
        s = "train:\t{}\ntest:\t{}".format(train_str, test_str)
        if self.name:
            s = "---- {}\n{}".format(self.name, s)
        return s

    def validate_shapes(self):
        for (X, Y, W) in [self.train_mats, self.test_mats]:
            N, D = X.shape
            D2, M = W.shape
            assert D == D2
            assert (N, M) == Y.shape

    def validate_hashes(self):
        assert self._hashes() == self.initial_hashes

    def validate(self, verbose=1, mse_thresh=1e-7, train=True, test=True):
        self.validate_shapes()
        self.validate_hashes()

        which_mats = []
        if train:
            which_mats.append(self.train_mats)
        if test:
            which_mats.append(self.test_mats)

        for (X, Y, W) in which_mats:
            Y_hat = X @ W
            diffs = Y - Y_hat
            mse = np.mean(diffs * diffs) / np.var(Y)
            if verbose > 0:
                print("mse: ", mse)
            assert mse < mse_thresh

    def _hashes(self):
        return {
            'X_train': self.X_train.std(),
            'Y_train': self.Y_train.std(),
            'W_train': self.W_train.std(),
            'X_test': self.X_test.std(),
            'Y_test': self.Y_test.std(),
            'W_test': self.W_test.std()
        }


# ================================================================ ECG

def _load_x_y_w_for_ar_model(data, window_len=4, verbose=1, N_train=-1,
                             # estimator='minlog'):
                             estimator='ridge'):

    # # TODO rm after debug
    # print("initial data shape: ", data.shape)
    # new_data = np.zeros((len(data), 4), dtype=data.dtype)
    # new_data[:, :3] = data
    # new_data[:, 3] = np.random.randn(len(data)) * np.std(data) * .01 + np.mean(data)
    # data = new_data

    data = data[1:] - data[:-1]  # predict 1st derivatives so nontrivial

    windows = window.sliding_window(
        data, ws=(window_len, data.shape[1]), ss=(1, 1))

    X = windows.reshape(windows.shape[0], -1)[:-1]
    Y = data[window_len:]

    # TODO rm
    # Y[1:] = Y[1:] - Y[:-1]  # predict differences, not raw values

    N = len(X)
    if N_train < data.shape[1]:
        N_train = N // 2  # TODO more flexible train/test split
    # N_test = N - N_train
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    # fit the autoregressive model (just a filter)
    # est = linear_model.LinearRegression(fit_intercept=False)
    if estimator == 'ridge':
        est = linear_model.Ridge(
            # alpha=.01*len(Y_train)*np.var(data), fit_intercept=False)
            # alpha=(.01 * np.var(data)), fit_intercept=False)
            alpha=(.1 * np.var(data)), fit_intercept=False)
        # est = linear_model.Lasso(
        #     # alpha=.001*np.sum(np.abs(Y_train)), fit_intercept=False)
        #     # alpha=1e-4*(Y_train * Y_train).sum(), fit_intercept=False)
        #     alpha=(1e-2 * Y_train.var()), fit_intercept=False)
        est.fit(X_train, Y_train)
        W = est.coef_.T
    else:
        W = algo.linear_regression_log_loss(X_train, Y_train)

    if verbose > 0:
        # print("ts ar model: data.shape: ", data.shape)
        # print("ts ar model: windows.shape: ", windows.shape)
        print("ts ar model: X shape: ", X.shape)
        print("ts ar model: Y shape: ", Y.shape)
        try:
            print("train r^2:", est.score(X_train, Y_train))
            print("test r^2:", est.score(X_test, Y_test))
        except UnboundLocalError:  # not using sklearn estimator
            pass
        diffs = Y[1:] - Y[:-1]
        # print("column variances of diffs", np.var(diffs, axis=0))
        # print("column variances of Y", np.var(Y, axis=0))
        # print("var(diffs), var(Y)", np.var(diffs), np.var(Y))
        print("var(diffs) / var(Y)", np.var(diffs) / np.var(Y))
        # print("coeffs: ")
        # for i in range(0, len(W), 10):
        #     print(W[i:(i + 10)])

        # Y_hat_train = est.predict(X_train)
        # Y_hat_test = est.predict(X_test)
        # print("Y_hat_train var / 1e3", np.var(Y_hat_train, axis=0) / 1e3)
        # print("Y_train var / 1e3", np.var(Y_train, axis=0) / 1e3)
        # print("Y_hat_test var / 1e3", np.var(Y_hat_test, axis=0) / 1e3)
        # print("Y_test var / 1e3", np.var(Y_test, axis=0) / 1e3)

        # import sys; sys.exit()

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


# def load_ecg_x_y_w_for_recording(recording, window_len=4):
#     return _load_x_y_w_for_ar_model(recording, window_len=window_len)

# @_memory.cache
# def load_ecg_recordings(limit_nhours=2):
#     generator = limit_nhours is not None and limit_nhours > 0
#     return sharee.load_recordings(
#         limit_nhours=limit_nhours, generator=generator)


# ------------------------------------------------ sharee

# @_memory.cache()  # caching is no faster than just recomputing with ridge
# @_memory.cache()
def load_sharee_x_y_w_for_recording_id(rec_id, window_len=4, limit_nhours=.5):
    rec = sharee.load_recording(rec_id, limit_nhours=limit_nhours)
    return _load_x_y_w_for_ar_model(rec, window_len=window_len)


def load_sharee_tasks(window_len=4, validate=False, **kwargs):
    rec_ids = sharee.load_recording_ids()
    for i, rec_id in enumerate(rec_ids):
        task = load_sharee_x_y_w_for_recording_id(
            rec_id, window_len=window_len)
        # task.info = {'rec_id: ', rec_id}
        task.name = rec_id
        if validate:
            print("validating ecg task {}/{}...".format(i + 1, len(rec_ids)))
            task.validate(mse_thresh=.25)  # normalized mse; >0 since lstsq
        yield task


# ------------------------------------------------ incart

# @_memory.cache()
def load_incart_x_y_w_for_recording_id(rec_id, window_len=4, limit_nhours=1):
    rec = incart.load_recording(rec_id, limit_nhours=limit_nhours)
    return _load_x_y_w_for_ar_model(rec, window_len=window_len)


def load_incart_tasks(window_len=4, validate=False, **kwargs):
    rec_ids = incart.load_recording_ids()
    for i, rec_id in enumerate(rec_ids):
        task = load_incart_x_y_w_for_recording_id(
            rec_id, window_len=window_len)
        task.name = rec_id
        if validate:
            print("validating ecg task {}/{}...".format(i + 1, len(rec_ids)))
            task.validate(mse_thresh=.25)  # normalized mse; >0 since lstsq
        yield task


# ------------------------------------------------ wrapper

def load_ecg_x_y_w_for_recording_id(*args, **kwargs):
    return load_incart_x_y_w_for_recording_id(*args, **kwargs)


def load_ecg_tasks(*args, **kwargs):
    return load_incart_tasks(*args, **kwargs)


# ================================================================ caltech

# def load_caltech_imgs(ntrain_classes=50, limit_per_class=10):
#     # imgs = caltech.load_caltech101(resample=None, crop=None)
#     (imgs, y), label2name = caltech.load_caltech101(
#         limit_per_class=limit_per_class)
#     # print(len(imgs))
#     # print(sum([img.size for img in imgs]))
#     # print("class counts: ", np.bincount(y))

#     # split by class; more relaxed than assuming you have examples from
#     # the same dataset/class you're later going to apply your filter to
#     train_idxs = np.where(y < ntrain_classes)[0]
#     test_idxs = np.where(y >= ntrain_classes)[0]
#     imgs_train = [imgs[i] for i in train_idxs]
#     imgs_test = [imgs[i] for i in test_idxs]

#     return imgs_train, imgs_test


@_memory.cache
def load_caltech_img_ids(ntrain_classes=50, limit_per_class_train=10,
                         limit_per_class_test=10, verbose=1):
    limit_per_class = max(limit_per_class_train, limit_per_class_test)
    (imgs, y), label2name = caltech.load_caltech101_ids(
        limit_per_class=limit_per_class)

    # split by class; more relaxed than assuming you have examples from
    # the same dataset/class you're later going to apply your filter to

    imgs_ids_train = []
    imgs_ids_test = []

    if verbose > 0:
        print("limiting ntrain per class to ", limit_per_class_train)
        print("limiting ntest per class to ", limit_per_class_test)

    # keep fewer idxs for train or test if requested
    if limit_per_class_train > 0:
        train_idxs = np.where(y < ntrain_classes)[0]
        if limit_per_class_train < limit_per_class:
            y_train = y[train_idxs]
            keep_idxs = []
            for c in np.unique(y_train):
                c_idxs = np.where(y_train == c)[0][:limit_per_class_train]
                keep_idxs += list(c_idxs)
            train_idxs = train_idxs[np.array(keep_idxs)]
        imgs_ids_train = [imgs[i] for i in train_idxs]

    if limit_per_class_test > 0:
        test_idxs = np.where(y >= ntrain_classes)[0]
        if limit_per_class_test < limit_per_class:
            y_test = y[test_idxs]
            keep_idxs = []
            for c in np.unique(y_test):
                c_idxs = np.where(y_test == c)[0][:limit_per_class_test]
                keep_idxs += list(c_idxs)
            test_idxs = test_idxs[np.array(keep_idxs)]
        imgs_ids_test = [imgs[i] for i in test_idxs]

    return imgs_ids_train, imgs_ids_test


def _load_dummy_caltech_filter_3x3(order='hwc'):
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
    filt_v = _lift_grayscale_filt_to_rgb(filt_v, order=order)
    filt_h = _lift_grayscale_filt_to_rgb(filt_h, order=order)

    return filt_v, filt_h


def load_filters_sobel_3x3(order='hwc'):
    filt_v = [[-1,  0, 1],
              [-2, 0, 2],
              [-1,  0, 1]]
    filt_v = np.array(filt_v, dtype=np.float32) / 2.
    return _lift_vert_filter_to_rgb_pair(filt_v, order=order)


def load_filters_sobel_5x5(order='hwc'):
    filt_v = [[-5,  -4, 0, 4,  5],
              [-8, -10, 0, 10, 8],
              [-10, 20, 0, 20, 10],
              [-8, -10, 0, 10, 8],
              [-5,  -4, 0, 4,  5]]
    filt_v = np.array(filt_v, dtype=np.float32) / 20.
    return _lift_vert_filter_to_rgb_pair(filt_v)


def load_filters_gaussian_3x3(order='hwc'):
    x = np.array([1, 2, 1])
    filt = (np.outer(x, x) / 16.).astype(np.float32)
    return [_lift_grayscale_filt_to_rgb(filt, order=order)]


def load_filters_gaussian_5x5(order='hwc'):
    x = np.array([1, 4, 6, 4, 1])
    filt = (np.outer(x, x) / 256.).astype(np.float32)
    return [_lift_grayscale_filt_to_rgb(filt, order=order)]


def load_filters_sharpen_5x5(order='hwc'):
    # from https://en.wikipedia.org/wiki/Kernel_(image_processing)
    x = np.array([1, 4, 6, 4, 1])
    x = np.outer(x, x)
    x[2, 2] = -476
    filt = (x / -256.).astype(np.float32)
    return [_lift_grayscale_filt_to_rgb(filt, order=order)]


def load_filters_gaussian(
        order='hwc', shape=(5, 5), sigmas=(1, 2)):
    filts = []
    for sigma in sigmas:
        filt = np.zeros(shape, dtype=np.float32)
        coeff = 1. / (sigma * np.sqrt(2 * np.pi))
        scale = 1. / (2 * sigma * sigma)
        i0 = int(shape[0] - 1) // 2
        j0 = int(shape[1] - 1) // 2
        for i in range(shape[0]):
            for j in range(shape[1]):
                dist_sq = (i - i0)**2 + (j - j0)**2
                filt[i, j] = np.exp(-dist_sq * scale)
        filt *= coeff
        filts.append(filt)
    return [_lift_grayscale_filt_to_rgb(filt, order=order) for filt in filts]


def _filters_list_to_mat(filters):
    filters_flat = [filt.ravel() for filt in filters]
    return np.vstack(filters_flat).T


def caltech_x_y_for_img(img, filt_spatial_shape, filters_list=None, W=None,
                        strides=(1, 1), order='chw'):
    # extract and flatten windows into rows of X matrix
    if order == 'hwc':
        windows = window.extract_conv2d_windows(
            img, filt_shape=filt_spatial_shape, strides=strides)
        filt_sz = img.shape[-1] * np.prod(filt_spatial_shape)
        X = windows.reshape(-1, filt_sz)
    else:
        assert order == 'chw'
        assert img.shape[2] == 3  # assumes img in hwc order
        X_subs_list = []
        filt_spatial_sz = np.prod(filt_spatial_shape)
        for c in range(3):
            windows = window.extract_conv2d_windows(
                img[:, :, c][..., np.newaxis],
                filt_shape=filt_spatial_shape, strides=strides)
            X_subs_list.append(windows.reshape(-1, filt_spatial_sz))
        X = np.hstack(X_subs_list)

    assert X.max() <= 255
    assert X.min() >= 0

    W = _filters_list_to_mat(filters_list) if W is None else W

    return X, X @ W


@_memory.cache  # cache raw images to avoid IO, but dynamically produce windows
def _load_caltech_train_imgs(limit_per_class=10):
    train_ids, _ = load_caltech_img_ids(
        limit_per_class_train=limit_per_class, limit_per_class_test=0)
    imgs = [caltech.load_caltech_img(img_id) for img_id in train_ids]
    return imgs, train_ids


@_memory.cache  # cache raw images to avoid IO, but dynamically produce windows
def _load_caltech_test_imgs(limit_per_class=10):
    _, test_ids = load_caltech_img_ids(
        limit_per_class_train=0, limit_per_class_test=limit_per_class)
    imgs = [caltech.load_caltech_img(img_id) for img_id in test_ids]
    return imgs, test_ids


# def _load_caltech_train(filters, filt_spatial_shape):
# def _load_caltech_train(W, filt_spatial_shape, strides=(3, 3)):
# def _load_caltech_train(W, filt_spatial_shape, strides=(1, 1), order='chw',
def _load_caltech_train(W, filt_spatial_shape, strides=(2, 2), order='chw',
                        limit_ntrain=-1, limit_per_class=10):
    train_imgs, _ = _load_caltech_train_imgs(limit_per_class=limit_per_class)

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

    train_mats = [caltech_x_y_for_img(img, W=W, strides=strides, order=order,
                                      filt_spatial_shape=filt_spatial_shape)
                  for img in train_imgs]
    Xs, Ys = list(zip(*train_mats))
    X_train = np.vstack(Xs)
    Y_train = np.vstack(Ys)

    if limit_ntrain is not None and limit_ntrain > 0:
        limit_ntrain = int(limit_ntrain)
        # X_train = X_train[:limit_ntrain]
        # Y_train = Y_train[:limit_ntrain]
        X_train = X_train[-limit_ntrain:]
        Y_train = Y_train[-limit_ntrain:]

    print("caltech training shape: ", X_train.shape, Y_train.shape)

    return X_train, Y_train


def load_caltech_tasks(order='chw', limit_ntrain=-1,
                       limit_ntest=-1, validate=False, filt='sobel',
                       limit_per_class_train=1,
                       limit_per_class_test=10):
    if filt == 'sobel':
        filters = load_filters_sobel_3x3(order=order)
        # filt_spatial_shape = (3, 3)
    elif filt == 'gauss3x3':
        filters = load_filters_gaussian_3x3(order=order)
        # filt_spatial_shape = (3, 3)
    elif filt == 'gauss5x5':
        filters = load_filters_gaussian_5x5(order=order)
        # filt_spatial_shape = (5, 5)
    elif filt == 'sharpen5x5':
        filters = load_filters_sharpen_5x5(order=order)
        # filt_spatial_shape = (5, 5)
    else:
        assert filt == 'dog5x5'
        filters = load_filters_gaussian(order=order, shape=(5, 5))
        # filt_spatial_shape = (5, 5)

    if order == 'chw':
        filt_spatial_shape = filters[0].shape[-2:]
    else:
        assert order == 'hwc'
        filt_spatial_shape = filters[0].shape[:2]

    W = _filters_list_to_mat(filters)
    X_train, Y_train = _load_caltech_train(
            W=W, filt_spatial_shape=filt_spatial_shape, order=order,
            limit_ntrain=limit_ntrain, limit_per_class=limit_per_class_train)

    test_imgs, test_ids = _load_caltech_test_imgs(
        limit_per_class=limit_per_class_test)

    # print("caltech tasks stats:")
    # print("X train shape: ", X_train.shape)
    # print("X train nbytes: ", X_train.nbytes)
    # print("Y train shape: ", Y_train.shape)
    # print("Y train nbytes: ", Y_train.nbytes)
    # # print("type(test_imgs)", type(test_imgs))
    # print("len(test_imgs)", len(test_imgs))

    # _, test_ids = load_caltech_img_ids(limit_per_class)
    # for i, _ in enumerate(test_imgs):
    for i, img in enumerate(test_imgs):
        # if i < 2: # TODO rm after debug
        #     continue

        # img = img1.copy() if i % 2 else img0.copy()
        # img = img1

        X_test, Y_test = caltech_x_y_for_img(
            img, filt_spatial_shape=filt_spatial_shape, W=W, order=order)
        name = f'Caltech {i} ({os.path.dirname(test_ids[i]).split("/")[-1]})'
        task = MatmulTask(X_train=X_train, Y_train=Y_train, W_train=W,
                          X_test=X_test, Y_test=Y_test, W_test=W,
                          name=name)
        task.info['problem'] = filt

        # print(f"task {task.name} matrix hashes:")
        # import pprint
        # pprint.pprint(task.hashes())

        if limit_ntest is not None and limit_ntest > 0:
            limit_ntest = int(limit_ntest)
            task.X_test = task.X_test[:limit_ntest]
            task.Y_test = task.Y_test[:limit_ntest]

        # task.info = {'task_id: ', i}
        # task.name = str(i)
        if validate:
            print("validating caltech task {}/{}...".format(
                i + 1, len(test_imgs)))
            print("X_train.std()", X_train.std())
            print("Y_train.std()", Y_train.std())
            print("X_test.std()", X_test.std())
            print("Y_test.std()", Y_test.std())
            task.validate()
        # print("about to yield task with name: ", task.name)
        yield task

    # print("exiting at load_caltech_tasks()")
    # import sys; sys.exit()


def test_caltech_loading():
    imgs_train, imgs_test = _load_caltech_test_imgs()
    filt = _load_dummy_caltech_filter_3x3()
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

def load_cifar10_tasks():
    SOFTMAX_INPUTS_TRAIN_PATH = 'cifar10_softmax_inputs_train.npy'
    SOFTMAX_OUTPUTS_TRAIN_PATH = 'cifar10_softmax_outputs_train.npy'
    SOFTMAX_INPUTS_TEST_PATH = 'cifar10_softmax_inputs_test.npy'
    SOFTMAX_OUTPUTS_TEST_PATH = 'cifar10_softmax_outputs_test.npy'
    SOFTMAX_W_PATH = 'cifar10_softmax_W.npy'
    SOFTMAX_B_PATH = 'cifar10_softmax_b.npy'
    LABELS_TRAIN_PATH = 'cifar10_labels_train.npy'
    LABELS_TEST_PATH = 'cifar10_labels_test.npy'

    def load_mat(fname):
        fpath = os.path.join(CIFAR10_DIR, fname)
        return np.load(fpath)

    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)
    lbls_train = load_mat(LABELS_TRAIN_PATH).ravel()
    lbls_test = load_mat(LABELS_TEST_PATH).ravel()

    # we aren't going to store or approximate the biases, so just subtract
    # off their contributions at the start
    Y_train -= b
    Y_test -= b

    # # TODO rm all this after debug
    # logits_test = Y_test + b
    # print("logits_test.shape", logits_test.shape)
    # print("lbls_test.shape", lbls_test.shape)
    # lbls_hat_test = np.argmax(Y_test, axis=1)
    # print("lbls_hat_test.shape", lbls_hat_test.shape)
    # acc = np.mean(lbls_hat_test.ravel() == lbls_test.ravel())
    # print("Y_test: ", Y_test[:10])
    # print("Y_train head: ", Y_train[:10])
    # print("Y_train tail: ", Y_train[-10:])
    # print("b:\n", b)
    # # print("lbls hat test:")
    # # print(lbls_hat_test[:100])
    # # print("lbls test:")
    # # print(lbls_test[:100])
    # print("lbls train:")
    # print(lbls_train[:100])
    # print("acc: ", acc)

    info = {'problem': 'softmax', 'biases': b,
            'lbls_train': lbls_train, 'lbls_test': lbls_test}

    return [MatmulTask(X_train, Y_train, X_test, Y_test, W,
                       name='CIFAR-10 Softmax', info=info)]


def load_cifar100_tasks():
    SOFTMAX_INPUTS_TRAIN_PATH = 'cifar100_softmax_inputs_train.npy'
    SOFTMAX_OUTPUTS_TRAIN_PATH = 'cifar100_softmax_outputs_train.npy'
    SOFTMAX_INPUTS_TEST_PATH = 'cifar100_softmax_inputs_test.npy'
    SOFTMAX_OUTPUTS_TEST_PATH = 'cifar100_softmax_outputs_test.npy'
    SOFTMAX_W_PATH = 'cifar100_softmax_W.npy'
    SOFTMAX_B_PATH = 'cifar100_softmax_b.npy'
    LABELS_TRAIN_PATH = 'cifar100_labels_train.npy'
    LABELS_TEST_PATH = 'cifar100_labels_test.npy'

    def load_mat(fname):
        fpath = os.path.join(CIFAR100_DIR, fname)
        return np.load(fpath)

    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)
    lbls_train = load_mat(LABELS_TRAIN_PATH).ravel()
    lbls_test = load_mat(LABELS_TEST_PATH).ravel()

    # we aren't going to store or approximate the biases, so just subtract
    # off their contributions at the start
    Y_train -= b
    Y_test -= b

    # # TODO rm all this after debug
    # logits_test = Y_test + b
    # print("logits_test.shape", logits_test.shape)
    # print("lbls_test.shape", lbls_test.shape)
    # lbls_hat_test = np.argmax(Y_test, axis=1)
    # print("lbls_hat_test.shape", lbls_hat_test.shape)
    # acc = np.mean(lbls_hat_test.ravel() == lbls_test.ravel())
    # print("Y_test: ", Y_test[:10])
    # print("Y_train head: ", Y_train[:10])
    # print("Y_train tail: ", Y_train[-10:])
    # print("b:\n", b)
    # # print("lbls hat test:")
    # # print(lbls_hat_test[:100])
    # # print("lbls test:")
    # # print(lbls_test[:100])
    # print("lbls train:")
    # print(lbls_train[:100].ravel())
    # print("acc: ", acc)

    info = {'problem': 'softmax', 'biases': b,
            'lbls_train': lbls_train, 'lbls_test': lbls_test}

    return [MatmulTask(X_train, Y_train, X_test, Y_test, W,
                       name='CIFAR-100 Softmax', info=info)]


def load_cifar_tasks():
    return load_cifar10_tasks() + load_cifar100_tasks()


# ================================================================ ucr

def _learn_neighbor_compression_W_info(X, lbls, k):
    centroids, lbls_centroids = algo.stochastic_neighbor_compression(
        X, lbls, k)
    extra_info = {'lbls_centroids': lbls_centroids}
    return centroids.T, extra_info


def _learn_softmax_W_info(X, lbls):
    est = linear_model.LogisticRegression(
        # raise max iters from 100 to avoid convergence messages
        fit_intercept=False, solver='lbfgs', max_iter=200)
    est.fit(X, lbls)
    nclasses, _ = est.coef_.shape
    return est.coef_.T, {'biases': np.zeros_like(nclasses, dtype=np.float32)}


@_memory.cache
def _load_ucr_task_for_dset(
        dset_name, D=320, k=128, min_train_sz=-1, use_test_sz=-1,
        problem='rbf', verbose=1):

    dset = ucr.UCRDataset(dset_name)
    if min_train_sz is None or min_train_sz < k:
        min_train_sz = k
    if use_test_sz is None or use_test_sz < 1:
        use_test_sz = len(dset.X_test)

    if verbose > 0:
        print(f"----- loading task for UCR dataset: {dset.name}")

    nclasses = len(np.unique(dset.y_test))
    ntrain = len(dset.X_train)
    ntest = len(dset.X_test)
    if nclasses > k:
        if verbose > 0:
            print(f"returning None because " +
                  f"nclasses={nclasses} > k={k}")
        return None  # some class will have no centroids
    if ntrain < min_train_sz:
        if verbose > 0:
            print(f"returning None because " +
                  f"num_train={ntrain} < min_train_sz={min_train_sz}")
        return None
    if ntest < use_test_sz:
        if verbose > 0:
            print(f"returning None because " +
                  f"num_test={ntest} < min_test_sz={use_test_sz}")
        return None

    X_train = dset.X_train
    X_test = dset.X_test[:use_test_sz]
    dset.y_test = dset.y_test[:use_test_sz]
    X_train = signal.resample(X_train, D, axis=1).astype(np.float32)
    X_test = signal.resample(X_test, D, axis=1).astype(np.float32)

    info = {'problem': problem, 'lbls_train': dset.y_train,
            'lbls_test': dset.y_test}
    if problem in ('1nn', 'rbf'):
        print(f"compressing training set for dset: {dset.name}")
        W, extra_info = _learn_neighbor_compression_W_info(
            X_train, dset.y_train, k)
    elif problem == 'softmax':
        W, extra_info = _learn_softmax_W_info(
            X_train, dset.y_train)
    else:
        raise ValueError(f"Unrecognized problem '{problem}'")

    Y_train = X_train @ W
    Y_test = X_test @ W
    info.update(extra_info)

    return [MatmulTask(X_train, Y_train, X_test, Y_test, W,
                       name=f'ucr {dset.name} k={k}', info=info)]


def load_ucr_tasks(limit_ntasks=-1, k=128, **kwargs):
    all_tasks = []
    df = ucr.load_ucr_dset_stats()
    name2acc = dict(zip(df['Name'], df['l2-1nn-acc']))

    for dset_name in ucr.all_ucr_dataset_dirs():
        orig_acc = name2acc[os.path.basename(dset_name)]
        tasks = _load_ucr_task_for_dset(dset_name, k=k, **kwargs)
        if tasks is not None:
            for task in tasks:
                task.info['acc-1nn-raw'] = orig_acc
            all_tasks += tasks
        # else:
        #     print("got None instead of tasks for dset: ", dset_name)
        if ((limit_ntasks is not None) and (limit_ntasks > 0) and
                (len(all_tasks) >= limit_ntasks)):
            all_tasks = all_tasks[:limit_ntasks]
            break
    return all_tasks


# ================================================================ main

def test_caltech_tasks():
    for _ in load_caltech_tasks(validate=True):
        pass  # need to loop thru since it's a generator


def test_ecg_tasks():
    # for _ in load_ecg_tasks(validate=True):
    for i, _ in enumerate(load_ecg_tasks(validate=False)):
        print("loaded ecg task {}/{}".format(i + 1, sharee.NUM_RECORDINGS))


def test_cifar_tasks():
    task = load_cifar10_tasks()[0]
    print(task)
    task.validate()
    task = load_cifar100_tasks()[0]
    print(task)
    task.validate()


def main():
    np.set_printoptions(formatter={'float': lambda f: "{:.3}".format(f)})

    train_ids, test_ids = load_caltech_img_ids()
    print("number of uniq train ids: ", len(np.unique(train_ids)))
    print("number of uniq test ids: ", len(np.unique(test_ids)))

    for i, task in enumerate(load_caltech_tasks(validate=True)):
        pass

    # test_caltech_tasks()
    # test_cifar_tasks()
    # test_ecg_tasks()

    # load_cifar10_tasks()
    # load_cifar100_tasks()

    # print("number of ucr dirs:", len(list(ucr.all_ucr_dataset_dirs())))
    # tasks = load_ucr_tasks()
    # print("number of tasks meeting basic size criteria:", len(tasks))

    # print("number of caltech imgs: ", len(_load_caltech_test_imgs()))


if __name__ == '__main__':
    main()
