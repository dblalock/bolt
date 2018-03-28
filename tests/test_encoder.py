#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from sklearn.datasets import load_digits
import timeit

import bolt


# ================================================================ utils

def _dists_sq(X, q):
    diffs = X - q
    return np.sum(diffs * diffs, axis=-1)


def _dists_l1(X, q):
    diffs = np.abs(X - q)
    return np.sum(diffs, axis=-1)


def _element_size_bytes(x):
    return np.dtype(x.dtype).itemsize


def _corr(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    x = x.ravel() - np.mean(x)
    y = y.ravel() - np.mean(y)
    r = np.mean(x * y) / (np.std(x) * np.std(y))
    assert -1.00001 <= r <= 1.00001
    return r


def _sq_dists_to_vectors(X, queries, rowNorms=None, queryNorms=None):
    Q = queries.shape[0]

    mat_size = X.shape[0] * Q
    mat_size_bytes = _element_size_bytes(X[0] + queries[0])
    if mat_size_bytes > int(1e9):
        print("WARNING: _sq_dists_to_vectors: attempting to create a matrix"
              "of size {} ({}B)".format(mat_size, mat_size_bytes))

    if rowNorms is None:
        rowNorms = np.sum(X * X, axis=1, keepdims=True)

    if queryNorms is None:
        queryNorms = np.sum(queries * queries, axis=1)

    dotProds = np.dot(X, queries.T)
    return (-2 * dotProds) + rowNorms + queryNorms  # len(X) x len(queries)


def top_k_idxs(elements, k, smaller_better=True, axis=-1):
    if smaller_better:  # return indices of lowest elements
        which_nn = np.arange(k)
        return np.argpartition(elements, kth=which_nn, axis=axis)[:k]
    else:  # return indices of highest elements
        which_nn = (elements.shape[axis] - 1 - np.arange(k))[::-1]
        # print "elements.shape", elements.shape
        # print "using which_nn: ", which_nn
        return np.argpartition(elements, kth=which_nn, axis=axis)[-k:][::-1]


def _knn(X, Q, k=1000, print_every=5, block_sz=128):
    nqueries = Q.shape[0]
    nblocks = int(np.ceil(nqueries / float(block_sz)))

    truth = np.full((nqueries, k), -999, dtype=np.int32)

    if nqueries <= block_sz:
        dists = _sq_dists_to_vectors(Q, X)
        assert dists.shape == (Q.shape[0], X.shape[0])
        for i in range(nqueries):
            truth[i, :] = top_k_idxs(dists[i, :], k)
        return truth

    for b in range(nblocks):
        # recurse to fill in knn for each block
        start = b * block_sz
        end = min(start + block_sz, nqueries)
        rows = Q[start:end, :]
        truth[start:end, :] = _knn(X, rows, k=k, block_sz=block_sz)

        if b % print_every == 0:
            print("computing top k for query block " \
                  "{} (queries {}-{})...".format(b, start, end))

    assert np.all(truth != -999)
    return truth


def _create_randn_encoder(Ntrain=100, Ntest=20, D=64):
    enc = bolt.Encoder()
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
    [enc.transform(q) for q in Q]
    for k in [1, 3]:
        [enc.knn(q, k) for q in Q]


def _fmt_float(x):
    return '{}.'.format(int(x)) if x == int(x) else '{:.3f}'.format(x)


def _load_digits_X_Q(nqueries):
    X, _ = load_digits(return_X_y=True)
    return X[:-nqueries], X[-nqueries:]  # X, Q


def test_time_space_savings():  # mostly to verify readme code
    np.set_printoptions(formatter={'float_kind': _fmt_float})

    nqueries = 20
    X, Q = _load_digits_X_Q(nqueries)

    enc = bolt.Encoder(accuracy='lowest', reduction=bolt.Reductions.DOT_PRODUCT)
    enc.fit(X)

    # massive space savings
    print("original space usage: {}B".format(X.nbytes))  # 1777 * 64 * 8B = 909KB
    print("bolt space usage: {}B".format(enc.nbytes))  # 1777 * 2B = 3.55KB

    # massive time savings (~10x here, but often >100x on larger datasets
    # with less Python overhead; see the Bolt paper)
    t_np = timeit.Timer(lambda: [np.dot(X, q) for q in Q]).timeit(5)  # ~8ms
    t_bolt = timeit.Timer(lambda: [enc.transform(q) for q in Q]).timeit(5)  # ~800us
    print("Numpy / BLAS time, Bolt time: {:.3f}ms, {:.3f}ms".format(
        t_np * 1000, t_bolt * 1000))


def test_unquantize():
    X, Q = _load_digits_X_Q(nqueries=20)
    enc = bolt.Encoder('dot', accuracy='high').fit(X)

    dots_true = [np.dot(X, q) for q in Q]
    dots_bolt = [enc.transform(q, unquantize=True) for q in Q]

    diffs = [true_vals - bolt_vals
             for true_vals, bolt_vals in zip(dots_true, dots_bolt)]
    mse = np.mean([np.mean(diff*diff) for diff in diffs])
    var = np.mean([np.var(true_vals) for true_vals in dots_true])
    print("dot product unquantize mse / variance: ", mse / var)
    assert (mse / var) < .01

    # print "true, bolt dot prods"
    # print dots_true[0][:20].astype(np.int32)
    # print dots_bolt[0][:20].astype(np.int32)

    enc = bolt.Encoder('l2', accuracy='high').fit(X)
    dists_true = [_dists_sq(X, q) for q in Q]
    dists_bolt = [enc.transform(q, unquantize=True) for q in Q]

    diffs = [true_vals - bolt_vals
             for true_vals, bolt_vals in zip(dists_true, dists_bolt)]
    mse = np.mean([np.mean(diff*diff) for diff in diffs])
    var = np.mean([np.var(true_vals) for true_vals in dots_true])
    print("squared l2 unquantize mse / variance: ", mse / var)
    assert (mse / var) < .01


def test_basic():
    # np.set_printoptions(precision=3)
    np.set_printoptions(formatter={'float_kind': _fmt_float})

    nqueries = 20
    # nqueries = 10
    # nqueries = 3
    X, Q = _load_digits_X_Q(nqueries)

    # TODO rm this block
    # shift = 100.
    # shift = 100
    # scaleby = 1.
    # scaleby = 3.5  # acc goes to **** at accelerating rate as this gets larger...
    # scaleby = 4
    # scaleby = 1.0
    # X, Q = X + shift, Q + shift
    # X, Q = X * scaleby, Q * scaleby
    # X = X[:200]
    # X = X[:50]
    # X = X[:20]

    # X, _ = load_digits(return_X_y=True)
    # Q = X[-nqueries:]
    # X = X[:-nqueries]

    # print "X.shape", X.shape
    # print "X nbytes", X.nbytes

    # ------------------------------------------------ squared l2

    enc = bolt.Encoder(accuracy='low', reduction=bolt.Reductions.SQUARED_EUCLIDEAN)
    enc.fit(X)

    l2_corrs = np.empty(len(Q))
    for i, q in enumerate(Q):
        l2_true = _dists_sq(X, q).astype(np.int)
        l2_bolt = enc.transform(q)
        l2_corrs[i] = _corr(l2_true, l2_bolt)
        if i == nqueries - 1:
            print("l2 true: ", l2_true)
            print("l2 bolt: ", l2_bolt)
            print("corr: ", l2_corrs[i])

    mean_l2 = np.mean(l2_corrs)
    std_l2 = np.std(l2_corrs)
    assert mean_l2 > .95
    print("--> squared l2 dist correlation: {} +/- {}".format(mean_l2, std_l2))

    # return

    # ------------------------------------------------ dot product

    enc = bolt.Encoder(accuracy='low', reduction=bolt.Reductions.DOT_PRODUCT)
    enc.fit(X)

    dot_corrs = np.empty(nqueries)
    for i, q in enumerate(Q):
        dots_true = np.dot(X, q)
        dots_bolt = enc.transform(q)
        dot_corrs[i] = _corr(dots_true, dots_bolt)

    mean_dot = np.mean(dot_corrs)
    std_dot = np.std(dot_corrs)
    assert mean_dot > .95
    print("--> dot product correlation: {} +/- {}".format(mean_dot, std_dot))

    # ------------------------------------------------ l2 knn

    enc = bolt.Encoder(accuracy='low', reduction='l2')
    enc.fit(X)

    k_bolt = 10  # tell bolt to search for true knn
    k_true = 10  # compute this many true neighbors
    true_knn = _knn(X, Q, k_true)
    bolt_knn = [enc.knn(q, k_bolt) for q in Q]

    contained = np.empty((nqueries, k_bolt), dtype=np.bool)
    for i in range(nqueries):
        true_neighbors = true_knn[i]
        bolt_neighbors = bolt_knn[i]
        for j in range(k_bolt):
            contained[i, j] = bolt_neighbors[j] in true_neighbors

    precision = np.mean(contained)
    print("--> l2 knn precision@{}: {}".format(k_bolt, precision))
    assert precision > .6

    # # print "true_knn, bolt_knn:"
    # # print true_knn[:20, :20]
    # # print bolt_knn[:20]

    # ------------------------------------------------ dot knn

    enc = bolt.Encoder(accuracy='low', reduction='dot')
    # enc = bolt.Encoder(accuracy='high', reduction='dot')
    enc.fit(X)

    k_bolt = 10  # tell bolt to search for true knn
    k_true = 10  # compute this many true neighbors
    true_dists = np.dot(X, Q.T)
    # true_dists = [np.dot(X, q) for q in Q]
    true_knn = np.empty((nqueries, k_true), dtype=np.int64)
    for i in range(nqueries):
        true_knn[i, :] = top_k_idxs(
            true_dists[:, i], k_true, smaller_better=False)
    bolt_knn = [enc.knn(q, k_bolt) for q in Q]

    contained = np.empty((len(Q), k_bolt), dtype=np.bool)
    for i in range(len(Q)):
        true_neighbors = true_knn[i]
        # bolt_dists = enc.transform(Q[i])
        # bolt_neighbors = top_k_idxs(bolt_dists, k_bolt, smaller_better=True)
        bolt_neighbors = bolt_knn[i]  # TODO uncomment
        for j in range(k_bolt):
            contained[i, j] = bolt_neighbors[j] in true_neighbors

    precision = np.mean(contained)
    print("--> max inner product knn precision@{}: {}".format(
        k_bolt, precision))
    assert precision > .6

    # print("true_knn, bolt_knn:")
    # print(true_knn[:5])
    # print(bolt_knn[:5])


if __name__ == '__main__':
    test_basic()
