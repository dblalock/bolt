#!/usr/bin/env python

import itertools
import numpy as np
from sklearn import cluster
import types

import kmc2  # state-of-the-art kmeans initialization (as of NIPS 2016)

from joblib import Memory
_memory = Memory('.', verbose=0)


# ================================================================ misc

def is_dict(x):
    return isinstance(x, dict)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def as_list_or_tuple(x):
    return x if is_list_or_tuple(x) else [x]


def is_string(x):
    return isinstance(x, (str,))


def flatten_list_of_lists(l):
    return list(itertools.chain.from_iterable(l))


def element_size_bytes(x):
    return np.dtype(x.dtype).itemsize


# ================================================================ distance

def dists_sq(X, q):
    diffs = X - q
    return np.sum(diffs * diffs, axis=-1)


def dists_l1(X, q):
    diffs = np.abs(X - q)
    return np.sum(diffs, axis=-1)


def sq_dists_to_vectors(X, queries, rowNorms=None, queryNorms=None):
    Q = queries.shape[0]

    mat_size = X.shape[0] * Q
    mat_size_bytes = element_size_bytes(X[0] + queries[0])
    if mat_size_bytes > int(1e9):
        print("WARNING: sq_dists_to_vectors: attempting to create a matrix" \
            "of size {} ({}B)".format(mat_size, mat_size_bytes))

    if rowNorms is None:
        rowNorms = np.sum(X * X, axis=1, keepdims=True)

    if queryNorms is None:
        queryNorms = np.sum(queries * queries, axis=1)

    dotProds = np.dot(X, queries.T)
    return (-2 * dotProds) + rowNorms + queryNorms  # len(X) x len(queries)


def all_eq(x, y):
    if len(x) != len(y):
        return False
    if len(x) == 0:
        return True
    return np.max(np.abs(x - y)) < .001


def top_k_idxs(elements, k, smaller_better=True, axis=-1):
    if smaller_better:  # return indices of lowest elements
        which_nn = np.arange(k)
        return np.argpartition(elements, kth=which_nn, axis=axis)[:k]
    else:  # return indices of highest elements
        which_nn = len(elements) - 1 - np.arange(k)
        return np.argpartition(elements, kth=which_nn, axis=axis)[-k:][::-1]


def compute_true_knn(X, Q, k=1000, print_every=5, block_sz=128):
    nqueries = Q.shape[0]
    nblocks = int(np.ceil(nqueries / float(block_sz)))

    truth = np.full((nqueries, k), -999, dtype=np.int32)

    if nqueries <= block_sz:
        dists = sq_dists_to_vectors(Q, X)
        assert dists.shape == (Q.shape[0], X.shape[0])
        for i in range(nqueries):
            truth[i, :] = top_k_idxs(dists[i, :], k)
            # truth[i, :] = top_k_idxs(dists[:, i], k)
        return truth

    for b in range(nblocks):
        # recurse to fill in knn for each block
        start = b * block_sz
        end = min(start + block_sz, nqueries)
        rows = Q[start:end, :]
        truth[start:end, :] = compute_true_knn(X, rows, k=k, block_sz=block_sz)

        if b % print_every == 0:
            print("computing top k for query block " \
                "{} (queries {}-{})...".format(b, start, end))

    # for i in range(nqueries):
    #     if i % print_every == 0:
    #         print "computing top k for query {}...".format(i)
    #     truth[i, :] = top_k_idxs(dists[i, :], k)
    print("done")

    assert np.all(truth != -999)
    return truth


def knn(X, q, k, dist_func=dists_sq):
    dists = dist_func(X, q)
    idxs = top_k_idxs(dists, k)
    return idxs, dists[idxs]


@_memory.cache
def kmeans(X, k, max_iter=16, init='kmc2'):
    X = X.astype(np.float32)

    # if k is huge, initialize centers with cartesian product of centroids
    # in two subspaces
    sqrt_k = int(np.sqrt(k) + .5)
    if k > 256 and sqrt_k ** 2 == k and init == 'subspaces':
        print("kmeans: clustering in subspaces first; k, sqrt(k) =" \
            " {}, {}".format(k, sqrt_k))
        _, D = X.shape
        centroids0, _ = kmeans(X[:, :D/2], sqrt_k, max_iter=1)
        centroids1, _ = kmeans(X[:, D/2:], sqrt_k, max_iter=1)
        seeds = np.empty((k, D), dtype=np.float32)
        for i in range(sqrt_k):
            for j in range(sqrt_k):
                row = i * sqrt_k + j
                seeds[row, :D/2] = centroids0[i]
                seeds[row, D/2:] = centroids1[j]
    elif init == 'kmc2':
        seeds = kmc2.kmc2(X, k).astype(np.float32)
    else:
        raise ValueError("init parameter must be one of {'kmc2', 'subspaces'}")

    estimator = cluster.MiniBatchKMeans(k, init=seeds, max_iter=max_iter, n_init=1).fit(X)
    return estimator.cluster_centers_, estimator.labels_


def orthonormalize_rows(A):
    Q, R = np.linalg.qr(A.T)
    return Q.T


def random_rotation(D):
    rows = np.random.randn(D, D)
    return orthonormalize_rows(rows)


def hamming_dist(v1, v2):
    return np.count_nonzero(v1 != v2)


def hamming_dists(X, q):
    return np.array([hamming_dist(row, q) for row in X])


if __name__ == '__main__':

    a = np.random.randn(10)
    sort_idxs = np.argsort(a)[::-1]
    print(a)
    print(top_k_idxs(a, 3, smaller_better=False))
    print(sort_idxs[:3])
