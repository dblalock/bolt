#!/usr/bin/env python

import numpy as np
from scipy.stats import pearsonr as corr
from sklearn.datasets import load_digits

import bolt


# ================================================================ utils

def _dists_sq(X, q):
    diffs = X - q
    return np.sum(diffs * diffs, axis=-1)


def _dists_l1(X, q):
    diffs = np.abs(X - q)
    return np.sum(diffs, axis=-1)


def _create_randn_encoder(nbytes=16, Ntrain=100, Ntest=20, D=64):
    enc = bolt.Encoder(nbytes=16)
    X_train = np.random.randn(Ntrain, D)
    X_test = np.random.randn(Ntest, D)
    enc.fit(X_train, just_train=True)
    enc.set_data(X_test)
    return enc


# ================================================================ tests

def test_smoketest():
    """Test that `bolt.Encoder`'s methods don't crash"""

    D = 64
    enc = _create_randn_encoder(D=D)

    Nqueries = 5
    Q = np.random.randn(Nqueries, D)
    [enc.dot(q) for q in Q]
    [enc.dists_sq(q) for q in Q]
    for k in [1, 3]:
        [enc.knn_l2(q, k) for q in Q]
        [enc.knn_dot(q, k) for q in Q]

    # assert False  # yep, this makes it fail, so actually running this test


def test_basic():
    # np.set_printoptions(precision=3)
    np.set_printoptions(formatter={'float_kind': lambda x: '{:.3f}'.format(x)})

    X, _ = load_digits(return_X_y=True)
    X = X[:, :16]

    num_queries = 20
    Q = X[-num_queries:]
    X = X[:-num_queries]

    enc = bolt.Encoder(nbytes=2, reduction=bolt.Reductions.SQUARED_EUCLIDEAN)
    enc.fit(X)

    # dot_corrs = np.empty(len(Q))
    l2_corrs = np.empty(len(Q))
    # for i, q in enumerate(Q[:1]):
    for i, q in enumerate(Q):
        # dots_true = np.dot(X, q)
        # dots_bolt = enc.dot(q)
        # dot_corrs[i] = corr(dots_true, dots_bolt)[0]

        # print "dots true, dots bolt:"
        # print dots_true
        # print dots_bolt

        # l2_true = _dists_sq(X, q)
        l2_true = _dists_sq(X, q).astype(np.int)
        l2_bolt = enc.dists_sq(q)

        # l2_bolt = l2_bolt[:len(l2_true)]  # bolt result includes padding

        # l2_corrs[i] = corr(l2_true, l2_bolt)[0]  # TODO uncommment
        l2_corrs[i] = corr(l2_true[:20], l2_bolt[:20])[0]

        # print "dists sq true, dots bolt:"
        # print l2_true[:20]
        # print l2_bolt[:20]

    # print "mean dot product correlation, squared l2 dist correlation:"
    print "squared l2 dist correlation:"
    # print np.mean(dot_corrs[0])
    # print np.mean(l2_corrs[0])
    # print np.mean(dot_corrs)
    print np.mean(l2_corrs)


if __name__ == '__main__':
    test_basic()
