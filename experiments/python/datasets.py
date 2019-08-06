#!/bin/env python

import os
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

from joblib import Memory
_memory = Memory('.', verbose=1)

DATA_DIR = os.path.expanduser('~/Desktop/datasets/nn-search')
join = os.path.join


class Random:
    UNIFORM = 'uniform'
    GAUSS = 'gauss'
    WALK = 'walk'
    BLOBS = 'blobs'


class Gist:
    DIR = join(DATA_DIR, 'gist')
    TRAIN    = join(DIR, 'gist_train.npy')     # noqa
    TEST     = join(DIR, 'gist.npy')           # noqa
    TEST_100 = join(DIR, 'gist_100k.npy')      # noqa
    TEST_200 = join(DIR, 'gist_200k.npy')      # noqa
    QUERIES  = join(DIR, 'gist_queries.npy')   # noqa
    TRUTH    = join(DIR, 'gist_truth.npy')     # noqa


class Sift1M:
    DIR = join(DATA_DIR, 'sift1m')
    TRAIN    = join(DIR, 'sift_learn.npy')          # noqa
    TEST     = join(DIR, 'sift_base.npy')           # noqa
    TEST_100 = join(DIR, 'sift_100k.txt')           # noqa
    TEST_200 = join(DIR, 'sift_200k.txt')           # noqa
    QUERIES  = join(DIR, 'sift_queries.npy')        # noqa
    TRUTH    = join(DIR, 'sift_groundtruth.npy')    # noqa


class Sift10M:
    DIR = join(DATA_DIR, 'sift1b')
    # TRAIN    = join(DIR, 'big_ann_learn_10M.npy') # noqa
    TRAIN    = join(DIR, 'big_ann_learn_1M.npy')    # noqa  # TODO use 10M?
    TRAIN_1M = join(DIR, 'big_ann_learn_1M.npy')    # noqa
    TEST     = join(DIR, 'sift_10M.npy')            # noqa
    QUERIES  = join(DIR, 'sift_queries.npy')        # noqa
    TRUTH    = join(DIR, 'true_nn_idxs_10M.npy')    # noqa


class Deep1M:
    """256D PCA of convnet activations; see OTQ paper supporting
    webiste, http://sites.skoltech.ru/compvision/projects/aqtq/"""
    DIR         = join(DATA_DIR, 'deep1m')              # noqa
    TRAIN       = join(DIR, 'deep1M_learn.npy')         # noqa
    TEST        = join(DIR, 'deep1M_base.npy')          # noqa
    TEST_100    = join(DIR, 'deep1M_test_100k.npy')     # noqa
    QUERIES     = join(DIR, 'deep1M_queries.npy')       # noqa
    TRUTH_TRAIN = join(DIR, 'deep1M_truth_train.npy')   # noqa
    TRUTH       = join(DIR, 'deep1M_groundtruth.npy')   # noqa


class Convnet1M:
    DIR         = join(DATA_DIR, 'convnet1m')           # noqa
    TRAIN       = join(DIR, 'convnet_train.npy')        # noqa
    TEST        = join(DIR, 'convnet_test.npy')         # noqa
    TEST_100    = join(DIR, 'convnet_test_100k.npy')    # noqa
    QUERIES     = join(DIR, 'convnet_queries.npy')      # noqa
    TRUTH_TRAIN = join(DIR, 'truth_train.npy')          # noqa
    TRUTH       = join(DIR, 'truth_test.npy')           # noqa


class Mnist:
    # following other papers (eg, "revisiting additive quantization"),
    # use mnist test set as queries and training set as database
    DIR     = join(DATA_DIR, 'mnist')                   # noqa
    TEST    = join(DIR, 'X_train.npy')                  # noqa
    QUERIES = join(DIR, 'X_test.npy')                   # noqa
    TRUTH   = join(DIR, 'truth_Q=test_X=train.npy')     # noqa


class LabelMe:
    DIR     = join(DATA_DIR, 'labelme')         # noqa
    TRAIN   = join(DIR, 'labelme_train.npy')    # noqa
    TEST    = join(DIR, 'labelme_train.npy')    # noqa
    QUERIES = join(DIR, 'labelme_test.npy')     # noqa
    TRUTH   = join(DIR, 'labelme_truth.npy')    # noqa


class Glove:
    DIR      = join(DATA_DIR, 'glove')          # noqa
    TEST     = join(DIR, 'glove_test.npy')      # noqa
    TEST_100 = join(DIR, 'glove_100k.txt')      # noqa
    TEST_200 = join(DIR, 'glove_200k.txt')      # noqa
    QUERIES  = join(DIR, 'glove_queries.npy')   # noqa
    TRUTH    = join(DIR, 'glove_truth.npy')     # noqa


# note that we've only run the real experiments on the ones reported
# in the paper (i.e., no cherrypicking)
ALL_REAL_DATASETS = [
    Gist, Sift1M, Sift10M, Deep1M, Convnet1M, Mnist, LabelMe, Glove]


def load_file(fname, *args, **kwargs):
    if fname.split('.')[-1] == 'txt':
        return np.loadtxt(fname, *args, **kwargs)
    return np.load(fname, *args, **kwargs)


def extract_random_rows(X, how_many, remove_from_X=True):
    split_start = np.random.randint(len(X) - how_many - 1)
    split_end = split_start + how_many
    rows = np.copy(X[split_start:split_end])
    if remove_from_X:
        return np.vstack((X[:split_start], X[split_end:])), rows
    return X, rows


def _load_complete_dataset(which_dataset, num_queries=10):
    X_test = np.load(which_dataset.TEST)
    try:
        X_train = np.load(which_dataset.TRAIN)
        print("using separate test set!")
    except AttributeError:
        print("No training set found for dataset {}".format(str(which_dataset)))
        X_train = np.copy(X_test)
    try:
        Q = np.load(which_dataset.QUERIES)
    except AttributeError:
        assert num_queries > 1
        X_train, Q = extract_random_rows(X_train, how_many=num_queries)
    try:
        true_nn = np.load(which_dataset.TRUTH).astype(np.int)
    except AttributeError:
        true_nn = None

    return X_train, Q, X_test, true_nn


def _ground_truth_for_dataset(which_dataset):
    return None  # TODO


# XXX: not clear whether this function is correct in general, but works for
# 784D with the nzeros we get for 32 and 64 codebooks
def _insert_zeros(X, nzeros):
    N, D = X.shape
    D_new = D + nzeros
    X_new = np.zeros((N, D_new), dtype=X.dtype)

    step = int(D / (nzeros + 1)) - 1

    for i in range(nzeros):
        in_start = step * i
        in_end = in_start + step
        # out_start = in_start + i + 1
        out_start = (step + 1) * i
        out_end = out_start + step
        X_new[:, out_start:out_end] = X[:, in_start:in_end]

    # out_start = out_end
    # out_end += step

    out_end += 1  # account for the last 0
    remaining_len = D - in_end
    out_remaining_len = D_new - out_end
    # print "step", step
    # print "in_start, in_end", in_start, in_end
    # print "out_start, out_end", out_start, out_end
    # print "D, D_new", D, D_new
    # print "remaining_len, out_remaining_len", remaining_len, out_remaining_len
    assert remaining_len == out_remaining_len

    assert remaining_len >= 0
    if remaining_len:
        # X_new[:, out_end:out_end+remaining_len] = X[:, in_end:D]
        X_new[:, out_end:] = X[:, in_end:]

    assert np.array_equal(X[:, 0], X_new[:, 0])
    assert np.array_equal(X[:, -1], X_new[:, -1])

    return X_new


def ensure_num_cols_multiple_of(X, multiple_of):
    remainder = X.shape[1] % multiple_of
    if remainder > 0:
        return _insert_zeros(X, multiple_of - remainder)
    return X


# @_memory.cache  # uncomment to get same randomness each time
def load_dataset(which_dataset, N=-1, D=-1, norm_mean=False, norm_len=False,
                 num_queries=10, Ntrain=-1, D_multiple_of=-1):
    true_nn = None

    # randomly generated datasets
    if which_dataset == Random.UNIFORM:
        X_test = np.random.rand(N, D)
        X_train = np.random.rand(Ntrain, D) if Ntrain > 0 else X_test
        Q = np.random.rand(num_queries, D)
    elif which_dataset == Random.GAUSS:
        X_test = np.random.randn(N, D)
        X_train = np.random.randn(Ntrain, D) if Ntrain > 0 else X_test
        Q = np.random.randn(num_queries, D)
    elif which_dataset == Random.WALK:
        X_test = np.random.randn(N, D)
        X_test = np.cumsum(X_test, axis=1)
        X_train = np.copy(X_test)
        if Ntrain > 0:
            X_train = np.random.randn(Ntrain, D)
            X_train = np.cumsum(X_train)
        Q = np.random.randn(num_queries, D)
        Q = np.cumsum(Q, axis=-1)
    elif which_dataset == Random.BLOBS:
        # centers is D x D, and centers[i, j] = (i + j)
        centers = np.arange(D)
        centers = np.sum(np.meshgrid(centers, centers), axis=0)
        X_test, _ = make_blobs(n_samples=N, centers=centers)
        X_train = np.copy(X_test)
        if Ntrain > 0:
            X_train, _ = make_blobs(n_samples=Ntrain, centers=centers)
        Q, true_nn = make_blobs(n_samples=num_queries, centers=centers)

    # datasets that are just one block of a "real" dataset
    elif isinstance(which_dataset, str):
        # assert False # TODO rm after real experiments
        X_test = load_file(which_dataset)
        X_test, Q = extract_random_rows(X_test, how_many=num_queries)
        X_train = np.copy(X_test)
        true_nn = _ground_truth_for_dataset(which_dataset)

    # "real" datasets with predefined train, test, queries, truth
    elif which_dataset in ALL_REAL_DATASETS:
        X_train, Q, X_test, true_nn = _load_complete_dataset(
            which_dataset, num_queries=num_queries)

    else:
        raise ValueError("unrecognized dataset {}".format(which_dataset))

    N = X_test.shape[0] if N < 1 else N
    D = X_test.shape[1] if D < 1 else D
    X_test, X_train = np.copy(X_test)[:N, :D], X_train[:N, :D]
    Q = Q[:, :D] if len(Q.shape) > 1 else Q[:D]

    train_is_test = X_train.base is X_test or X_test.base is X_train
    train_test_equal = np.array_equal(X_train[:100], X_test[:100])
    train_test_same = train_is_test or train_test_equal

    if train_test_same:
        print("WARNING: Training data is also the test data!")

    if train_is_test:
        X_test = np.copy(X_test)

    if norm_mean:
        means = np.mean(X_train, axis=0)
        X_train -= means
        X_test -= means
        Q -= means
    if norm_len:
        X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)
        X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=-1, keepdims=True)

    # np.set_printoptions(precision=6)
    # print "start of Q:", Q[:5, :5]
    # print "start of X_test:", X_test[:5, :5]

    # TODO don't convert datasets that are originally uint8s to floats
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    # Q = np.squeeze(Q.astype(np.float32))
    Q = Q.astype(np.float32)

    if D_multiple_of > 1:
        X_train = ensure_num_cols_multiple_of(X_train, D_multiple_of)
        X_test = ensure_num_cols_multiple_of(X_test, D_multiple_of)
        Q = ensure_num_cols_multiple_of(Q, D_multiple_of)

    return X_train, Q, X_test, true_nn


def read_yael_vecs(path, c_contiguous=True, limit_rows=-1, dtype=None):
    dim = np.fromfile(path, dtype=np.int32, count=2)[0]
    print("vector length = {}".format(dim))

    if dtype is None:
        if 'fvecs' in path:
            dtype = np.float32
        elif 'ivecs' in path:
            dtype = np.int32
        elif 'bvecs' in path:
            dtype = np.uint8
        else:
            raise ValueError("couldn't infer dtype from path {}".format(path))
    itemsize = np.dtype(dtype).itemsize

    assert dim > 0
    assert itemsize in (1, 2, 4)

    cols_for_dim = 4 // itemsize
    row_size_bytes = 4 + dim * itemsize
    row_size_elems = row_size_bytes // itemsize
    limit = int(limit_rows) * row_size_elems if limit_rows > 0 else -1

    fv = np.fromfile(path, dtype=dtype, count=limit)
    fv = fv.reshape((-1, row_size_elems))

    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + path)

    fv = fv[:, cols_for_dim:]

    if c_contiguous:
        fv = fv.copy()
    return fv
