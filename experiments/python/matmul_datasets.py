#!/usr/bin/env python

# from future import absolute_import, division, print_function

import numpy as np
from sklearn import linear_model

from python.datasets import ampds, caltech
from python import window

from joblib import Memory
_memory = Memory('.', verbose=0)


def load_ampd_data_mat():
    # return ampds.all_power_recordings()[0].data
    return ampds.all_weather_recordings()[0].data
    # print("mat.shape", mat.shape)


@_memory.cache
def load_ampd_x_y_w(window_len=16, verbose=1):
    data = load_ampd_data_mat()
    windows = window.sliding_window(
        data, ws=(window_len, data.shape[1]), ss=(1, 1))

    # fit the autoregressive model (just a filter)
    X = windows.reshape(windows.shape[0], -1)[:-1]
    Y = windows[1:, -1, :]  # targets are last row of next window
    est = linear_model.LinearRegression(fit_intercept=False)
    est.fit(X, Y)
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


def load_caltech_imgs():
    # imgs = caltech.load_caltech101(resample=None, crop=None)
    (imgs, y), label2name = caltech.load_caltech101(
        resample=None, crop=None, limit_per_class=10)
    # print(len(imgs))
    # print(sum([img.size for img in imgs]))
    # print("class counts: ", np.bincount(y))
    return imgs


def load_caltech_filter_3x3(order='hwc'):
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


def caltech_x_y_for_img(img, filt):
    # W = filt.ravel()
    # img = img[np.newaxis, ...]
    windows = window.extract_conv2d_windows(img, filt_shape=filt.shape[1:])
    print(windows.shape)

    broadcast_filt = filt.reshape(1, 1, *filt.shape)
    dot_prods = np.sum(windows * broadcast_filt, axis=(2, 3, 4))

    X = windows.reshape(-1, filt.size)
    Y = dot_prods.reshape(len(X))

    return X, Y


def test_caltech_loading():
    imgs = load_caltech_imgs()
    filt = load_caltech_filter_3x3()

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


def main():
    # load_ampd_data_mat()
    # load_ampd_windows()
    # load_ampd_x_y_w()
    # load_caltech_imgs()
    test_caltech_loading()

    # load_caltech_filter_3x3()


if __name__ == '__main__':
    main()
