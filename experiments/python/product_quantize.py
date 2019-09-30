#!/usr/bin/env python

import time
import numpy as np

from .utils import kmeans, orthonormalize_rows, random_rotation

from joblib import Memory
_memory = Memory('.', verbose=0)


# ================================================================ PQ

@_memory.cache
def learn_pq(X, ncentroids, nsubvects, subvect_len, max_kmeans_iters=16):
    codebooks = np.empty((ncentroids, nsubvects, subvect_len))
    assignments = np.empty((X.shape[0], nsubvects), dtype=np.int)

    # print "codebooks shape: ", codebooks.shape

    for i in range(nsubvects):
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        centroids, labels = kmeans(X_in, ncentroids, max_iter=max_kmeans_iters)
        codebooks[:, i, :] = centroids
        assignments[:, i] = labels

    return codebooks, assignments  # [2**nbits x M x D/M], [N x M]


def reconstruct_X_pq(assignments, codebooks):
    """assignments: N x M ints; codebooks: 2**nbits x M x D/M floats"""
    _, M = assignments.shape
    subvect_len = codebooks.shape[2]

    assert assignments.shape[1] == codebooks.shape[1]

    D = M * subvect_len
    pointsCount = assignments.shape[0]
    points = np.zeros((pointsCount, D), dtype=np.float32)
    for i in range(M):
        subspace_start = subvect_len * i
        subspace_end = subspace_start + subvect_len
        subspace_codes = assignments[:, i]
        points[:, subspace_start:subspace_end] = codebooks[subspace_codes, i, :]
    return points


def _dists_elemwise_sq(x, q):
    diffs = x - q
    return diffs * diffs


def _dists_elemwise_l1(x, q):
    return np.abs(x - q)


def _encode_X_pq(X, codebooks, elemwise_dist_func=_dists_elemwise_sq):
    ncentroids, nsubvects, subvect_len = codebooks.shape

    assert X.shape[1] == (nsubvects * subvect_len)

    idxs = np.empty((X.shape[0], nsubvects), dtype=np.int)
    X = X.reshape((X.shape[0], nsubvects, subvect_len))
    for i, row in enumerate(X):
        row = row.reshape((1, nsubvects, subvect_len))
        dists = elemwise_dist_func(codebooks, row)
        dists = np.sum(dists, axis=2)
        idxs[i, :] = np.argmin(dists, axis=0)

    return idxs  # [N x nsubvects]


def compute_reconstruction_error(X, X_hat, subvect_len=-1):
    diffs = X - X_hat
    diffs_sq = diffs * diffs
    if subvect_len > 0:
        errs = []
        for i in range(0, diffs_sq.shape[1], subvect_len):
            errs_block = diffs_sq[:, i:i+subvect_len]
            errs.append(np.mean(errs_block))
        print("   errors in each block: {} ({})".format(
            np.array(errs), np.sum(errs)))

    X_bar = X - np.mean(X, axis=0)
    col_sses = np.sum(X_bar * X_bar, axis=0) + 1e-14
    tot_sse_using_mean = np.sum(col_sses)

    errors = np.mean(diffs_sq, axis=1)
    # variances = np.var(X, axis=1)
    # return np.mean(errors) / np.mean(variances)
    return np.mean(errors) / (tot_sse_using_mean / X_bar.size)


# ================================================================ Gaussian OPQ

# https://github.com/yahoo/lopq/blob/master/python/lopq/model.py; see
# https://github.com/yahoo/lopq/blob/master/LICENSE. For this function only:
#
# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0.
# See the LICENSE file associated with the project for terms.
#
@_memory.cache
def eigenvalue_allocation(num_buckets, eigenvalues, shuffle=False):
    """
    Compute a permutation of eigenvalues to balance variance accross buckets
    of dimensions.
    Described in section 3.2.4 in http://research.microsoft.com/pubs/187499/cvpr13opq.pdf
    Note, the following slides indicate this function will break when fed eigenvalues < 1
    without the scaling trick implemented below:
        https://www.robots.ox.ac.uk/~vgg/rg/slides/ge__cvpr2013__optimizedpq.pdf
    :param int num_buckets:
        the number of dimension buckets over which to allocate eigenvalues
    :param ndarray eigenvalues:
        a vector of eigenvalues
    :param bool shuffle:
        whether to randomly shuffle the order of resulting buckets
    :returns ndarray:
        a vector of indices by which to permute the eigenvectors
    """
    D = len(eigenvalues)
    dims_per_bucket = D // num_buckets
    eigenvalue_product = np.zeros(num_buckets, dtype=float)
    bucket_size = np.zeros(num_buckets, dtype=int)
    permutation = np.zeros((num_buckets, dims_per_bucket), dtype=int)

    # We first must scale the eigenvalues by dividing by their
    # smallets non-zero value to avoid problems with the algorithm
    # when eigenvalues are less than 1.
    min_non_zero_eigenvalue = np.min(np.abs(eigenvalues[np.nonzero(eigenvalues)]))
    eigenvalues = eigenvalues / min_non_zero_eigenvalue

    # this is not actually a requirement, but I'm curious about whether this
    # condition is ever violated
    if not np.all(eigenvalues > 0):
        print("WARNING: some eigenvalues were nonpositive")

    # Iterate eigenvalues in descending order
    sorted_inds = np.argsort(eigenvalues)[::-1]
    log_eigs = np.log2(abs(eigenvalues))
    for ind in sorted_inds:

        # Find eligible (not full) buckets
        eligible = (bucket_size < dims_per_bucket).nonzero()

        # Find eligible bucket with least eigenvalue product
        i = eigenvalue_product[eligible].argmin(0)
        bucket = eligible[0][i]

        # Update eigenvalue product for this bucket
        eigenvalue_product[bucket] = eigenvalue_product[bucket] + log_eigs[ind]

        # Store bucket assignment and update size
        permutation[bucket, bucket_size[bucket]] = ind
        bucket_size[bucket] += 1

    if shuffle:
        shuffle_idxs = np.arange(num_buckets, dtype=np.int)
        np.random.shuffle(shuffle_idxs)
        permutation = permutation[shuffle_idxs]

    # wow, these are within <1% of each other
    # print "opq eigenvalue log prods: ", eigenvalue_product

    return np.reshape(permutation, D)


def learn_opq_gaussian_rotation(X_train, ncodebooks, shuffle=False):
    means = np.mean(X_train, axis=0)
    cov = np.dot(X_train.T, X_train) - np.outer(means, means)
    eigenvals, eigenvects = np.linalg.eigh(cov)

    order_idxs = eigenvalue_allocation(ncodebooks, eigenvals, shuffle=shuffle)
    assert len(order_idxs) == X_train.shape[1]
    return eigenvects[:, order_idxs].T  # rows are projections


# ================================================================ OPQ

def _update_centroids_opq(X, assignments, ncentroids):  # [N x D], [N x M]
    nsubvects = assignments.shape[1]
    subvect_len = X.shape[1] // nsubvects

    assert X.shape[0] == assignments.shape[0]
    assert X.shape[1] % nsubvects == 0

    codebooks = np.zeros((ncentroids, nsubvects, subvect_len), dtype=np.float32)
    for i, row in enumerate(X):
        for m in range(nsubvects):
            start_col = m * subvect_len
            end_col = start_col + subvect_len
            codebooks[assignments[i, m], m, :] += row[start_col:end_col]

    for m in range(nsubvects):
        code_counts = np.bincount(assignments[:, m], minlength=ncentroids)
        codebooks[:, m] /= np.maximum(code_counts, 1).reshape((-1, 1))  # no div by 0

    return codebooks


class NumericalException(Exception):
    pass


def _debug_rotation(R):
    D = np.max(R.shape)
    identity = np.identity(D, dtype=np.float32)
    RtR = np.dot(R.T, R)

    R_det = np.linalg.det(RtR)
    print("determinant of R*R: ", R_det)
    R_trace = np.trace(RtR)
    print("trace of R*R, trace divided by D: {}, {}".format(R_trace, R_trace / D))
    off_diagonal_abs_mean = np.mean(np.abs(RtR - identity))
    print("mean(abs(off diagonals of R*R)): ", off_diagonal_abs_mean)

    if R_det < .999 or R_det > 1.001:
        raise NumericalException("Bad determinant")
    if R_trace < .999 * D or R_trace > 1.001 * D:
        raise NumericalException("Bad trace")
    if off_diagonal_abs_mean > .001:
        raise NumericalException("Bad off-diagonals")


def opq_rotate(X, R):  # so other code need not know what to transpose
    return np.dot(np.atleast_2d(X), R.T)


def opq_undo_rotate(X, R):  # so other code need not know what to transpose
    return np.dot(np.atleast_2d(X), R)


# @_memory.cache
def opq_initialize(X_train, ncodebooks, init='gauss'):
    X = X_train
    _, D = X.shape

    if init == 'gauss' or init == 'gauss_flat' or init == 'gauss_shuffle':
        permute = (init == 'gauss_shuffle')
        R = learn_opq_gaussian_rotation(X_train, ncodebooks, shuffle=permute)
        R = R.astype(np.float32)

        if init == 'gauss_flat':
            # assert R.shape[0] == R.shape[1]
            D = R.shape[1]
            d = D // ncodebooks
            assert d * ncodebooks == D  # same # of dims in each subspace
            local_r = random_rotation(int(d))
            tiled = np.zeros((D, D))
            for c in range(ncodebooks):
                start = c * d
                end = start + d
                tiled[start:end, start:end] = local_r

            R = np.dot(R, tiled)

        X_rotated = opq_rotate(X, R)
    elif init == 'identity':
        R = np.identity(D, dtype=np.float32)  # D x D
        X_rotated = X
    elif init == 'random':
        R = np.random.randn(D, D).astype(np.float32)
        R = orthonormalize_rows(R)
        X_rotated = opq_rotate(X, R)
    else:
        raise ValueError("Unrecognized initialization method: ".format(init))

    return X_rotated, R


# loosely based on:
# https://github.com/arbabenko/Quantizations/blob/master/opqCoding.py
@_memory.cache
def learn_opq(X_train, ncodebooks, codebook_bits=8, niters=10,
              initial_kmeans_iters=1, init='gauss', debug=False):
    """init in {'gauss', 'identity', 'random'}"""

    print("OPQ: Using init '{}'".format(init))

    t0 = time.time()

    X = X_train.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2**codebook_bits)
    subvect_len = D // ncodebooks

    assert D % subvect_len == 0  # equal number of dims for each codebook

    X_rotated, R = opq_initialize(X_train, ncodebooks=ncodebooks, init=init)

    # initialize codebooks by running kmeans on each rotated dim; this way,
    # setting niters=0 corresponds to normal PQ
    codebooks, assignments = learn_pq(X_rotated, ncentroids=ncentroids,
                                      nsubvects=ncodebooks,
                                      subvect_len=subvect_len,
                                      max_kmeans_iters=1)

    for it in np.arange(niters):
        # compute reconstruction errors
        X_hat = reconstruct_X_pq(assignments, codebooks)
        # err = compute_reconstruction_error(X_rotated, X_hat, subvect_len=subvect_len)
        err = compute_reconstruction_error(X_rotated, X_hat)
        print("---- OPQ {}x{}b iter {}: mse / variance = {:.5f}".format(
            ncodebooks, codebook_bits, it, err))

        # update rotation matrix based on reconstruction errors
        U, s, V = np.linalg.svd(np.dot(X_hat.T, X), full_matrices=False)
        R = np.dot(U, V)

        # update centroids using new rotation matrix
        X_rotated = opq_rotate(X, R)
        assignments = _encode_X_pq(X_rotated, codebooks)
        codebooks = _update_centroids_opq(X_rotated, assignments, ncentroids)

    X_hat = reconstruct_X_pq(assignments, codebooks)
    err = compute_reconstruction_error(X_rotated, X_hat)
    t = time.time() - t0
    print("---- OPQ {}x{}b final mse / variance = {:.5f} ({:.3f}s)".format(
        ncodebooks, codebook_bits, err, t))

    return codebooks, assignments, R


# ================================================================ Block OPQ

def bopq_rotate(X, rotations):
    X = np.atleast_2d(X)
    _, D = X.shape
    R_sz = len(rotations[0])
    nrots = int(D / R_sz)
    assert nrots == len(rotations)

    rot_starts = R_sz * np.arange(nrots)
    rot_ends = rot_starts + R_sz

    X_out = np.copy(X)
    for i, R in enumerate(rotations):
        start, end = rot_starts[i], rot_ends[i]
        X_out[:, start:end] = np.dot(X[:, start:end], R.T)

    return X_out


@_memory.cache  # opq with block diagonal rotations
def learn_bopq(X_train, ncodebooks, codebook_bits=4, niters=20,
               initial_kmeans_iters=1, R_sz=16, **sink):

    t0 = time.time()

    X = X_train.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2**codebook_bits)
    subvect_len = D // ncodebooks

    assert D % subvect_len == 0  # equal number of dims for each codebook

    # compute number of rotations and subspaces associated with each
    nrots = int(D / R_sz)
    rot_starts = R_sz * np.arange(nrots)
    rot_ends = rot_starts + R_sz

    # X_rotated, R = opq_initialize(X_train, ncodebooks=ncodebooks, init=init)
    X_rotated = X  # hardcode identity init # TODO allow others
    rotations = [np.eye(R_sz) for i in range(nrots)]

    # initialize codebooks by running kmeans on each rotated dim; this way,
    # setting niters=0 corresponds to normal PQ
    codebooks, assignments = learn_pq(X_rotated, ncentroids=ncentroids,
                                      nsubvects=ncodebooks,
                                      subvect_len=subvect_len,
                                      max_kmeans_iters=1)

    for it in np.arange(niters):
        # compute reconstruction errors
        X_hat = reconstruct_X_pq(assignments, codebooks)
        # err = compute_reconstruction_error(X_rotated, X_hat, subvect_len=subvect_len)
        err = compute_reconstruction_error(X_rotated, X_hat)
        print("---- BOPQ {} {}x{}b iter {}: mse / variance = {:.5f}".format(
            R_sz, ncodebooks, codebook_bits, it, err))

        rotations = []
        for i in range(nrots):
            start, end = rot_starts[i], rot_ends[i]

            X_sub = X[:, start:end]
            X_hat_sub = X_hat[:, start:end]

            # update rotation matrix based on reconstruction errors
            U, s, V = np.linalg.svd(np.dot(X_hat_sub.T, X_sub), full_matrices=False)
            R = np.dot(U, V)
            rotations.append(R)

            X_rotated[:, start:end] = np.dot(X_sub, R.T)

        # update assignments and codebooks based on new rotations
        assignments = _encode_X_pq(X_rotated, codebooks)
        codebooks = _update_centroids_opq(X_rotated, assignments, ncentroids)

    X_hat = reconstruct_X_pq(assignments, codebooks)
    err = compute_reconstruction_error(X_rotated, X_hat)
    t = time.time() - t0
    print("---- BOPQ {} {}x{}b final mse / variance = {:.5f} ({:.3f}s)".format(
        R_sz, ncodebooks, codebook_bits, err, t))

    return codebooks, assignments, rotations
