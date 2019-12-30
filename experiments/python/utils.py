#!/usr/bin/env python

import itertools
import numpy as np
from sklearn import cluster
from scipy import signal
# import types

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


def is_scalar_seq(x):
    try:
        [float(element) for element in x]
        return True
    except TypeError:
        return False


def as_scalar_seq(x):
    if is_scalar_seq(x):
        return x
    try:
        _ = float(x)
        return [x]
    except TypeError:
        raise TypeError("Couldn't convert value '{}' to sequence "
                        "of scalars".format(x))


def is_string(x):
    return isinstance(x, (str,))


def flatten_list_of_lists(l):
    return list(itertools.chain.from_iterable(l))


def element_size_bytes(x):
    return np.dtype(x.dtype).itemsize


def invert_permutation(permutation):
    return np.arange(len(permutation))[np.argsort(permutation)]


# ================================================================ image

def conv2d(img, filt, pad='same'):
    # assert pad in ('same',)  # TODO support valid
    # mode = 'constant'
    if len(img.shape) == 2:
        return signal.correlate2d(img, filt, mode=pad)

    # img is more than 2d; do a 2d conv for each channel and sum results
    assert len(img.shape) == 3
    out = np.zeros(img.shape[:2], dtype=np.float32)
    for c in range(img.shape[2]):
        f = filt[:, :, c] if len(filt.shape) == 3 else filt
        out += signal.correlate2d(img[:, :, c], f, mode=pad)
    return out


# def filter_img(img, filt):
#     out = conv2d(img, filt)
#     return out / np.max(out)


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
            print("computing top k for query block "
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
def kmeans(X, k, max_iter=16, init='kmc2', return_sse=False):
    X = X.astype(np.float32)

    # handle fewer nonzero rows than centroids (mostly just don't choke
    # if X all zeros, which happens when run in PQ with tiny subspaces)
    rowsums = X.sum(axis=1)
    nonzero_mask = rowsums != 0
    nnz_rows = np.sum(nonzero_mask)
    if nnz_rows < k:
        print("X.shape: ", X.shape)
        print("k: ", k)
        print("nnz_rows: ", nnz_rows)

        centroids = np.zeros((k, X.shape[1]), dtype=X.dtype)
        labels = np.full(X.shape[0], nnz_rows, dtype=np.int)
        if nnz_rows > 0:  # special case, because can't have slice of size 0
            # make a centroid out of each nonzero row, and assign only those
            # rows to that centroid; all other rows get assigned to next
            # centroid after those, which is all zeros
            centroids[nnz_rows] = X[nonzero_mask]
            labels[nonzero_mask] = np.arange(nnz_rows)
        if return_sse:
            return centroids, labels, 0
        return centroids, labels

    # if k is huge, initialize centers with cartesian product of centroids
    # in two subspaces
    sqrt_k = int(np.ceil(np.sqrt(k)))
    if k >= 16 and init == 'subspaces':
        print("kmeans: clustering in subspaces first; k, sqrt(k) ="
              " {}, {}".format(k, sqrt_k))
        _, D = X.shape
        centroids0, _ = kmeans(X[:, :D/2], sqrt_k, max_iter=1)
        centroids1, _ = kmeans(X[:, D/2:], sqrt_k, max_iter=1)
        seeds = np.empty((sqrt_k * sqrt_k, D), dtype=np.float32)
        for i in range(sqrt_k):
            for j in range(sqrt_k):
                row = i * sqrt_k + j
                seeds[row, :D/2] = centroids0[i]
                seeds[row, D/2:] = centroids1[j]
        seeds = seeds[:k]  # rounded up sqrt(k), so probably has extra rows
    elif init == 'kmc2':
        try:
            seeds = kmc2.kmc2(X, k).astype(np.float32)
        except ValueError:  # can happen if dist of 0 to centroid
            print("WARNING: couldn't use kmc2 initialization")
            seeds = 'k-means++' if k < max_iter else 'random'
    else:
        raise ValueError("init parameter must be one of {'kmc2', 'subspaces'}")

    est = cluster.MiniBatchKMeans(
        k, init=seeds, max_iter=max_iter, n_init=1).fit(X)
    if return_sse:
        return est.cluster_centers_, est.labels_, est.inertia_
    return est.cluster_centers_, est.labels_


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
