#!/usr/bin/env python

from __future__ import division, absolute_import

import abc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from . import product_quantize as pq
from . import subspaces as subs
from . import clusterize
from .utils import kmeans


# ================================================================ misc funcs

def dists_elemwise_sq(x, q):
    diffs = x - q
    return diffs * diffs


def dists_elemwise_l1(x, q):
    return np.abs(x - q)


def dists_elemwise_dot(x, q):
    return x * q


def extract_random_rows(X, how_many, remove_from_X=True):
    split_start = np.random.randint(len(X) - how_many - 1)
    split_end = split_start + how_many
    rows = np.copy(X[split_start:split_end])
    if remove_from_X:
        return np.vstack((X[:split_start], X[split_end:])), rows
    return X, rows


# XXX: not clear whether this function is correct in general, but works for
# 784D with the nzeros we get for 32 and 64 codebooks
def _insert_zeros(X, nzeros):
    N, D = X.shape
    D_new = D + nzeros
    X_new = np.zeros((N, D_new), dtype=X.dtype)
    print("attempting to insert {} zeros into X of shape {}".format(nzeros, X.shape))

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


# def ensure_num_cols_multiple_of(X, multiple_of, min_ncols=-1):
def ensure_num_cols_multiple_of(X, multiple_of):
    remainder = X.shape[1] % multiple_of
    if remainder > 0:
        return _insert_zeros(X, multiple_of - remainder)

        # # TODO rm and uncomment above after debug
        # add_ncols = multiple_of - remainder
        # new_ncols = X.shape[1] + add_ncols
        # new_X = np.zeros((X.shape[0], new_ncols), dtype=X.dtype)
        # new_X[:, :X.shape[1]] = X
        # return new_X

    return X


def _learn_best_quantization(luts):
    assert luts.ndim == 2  # luts can be a bunch of vstacked luts, but not 3D
    best_loss = np.inf
    best_alpha = None
    best_floors = None
    best_scale_by = None
    for alpha in [.001, .002, .005, .01, .02, .05, .1]:
        # alpha_pct = int(100 * alpha)
        alpha_pct = 100 * alpha

        # compute quantized luts this alpha would yield
        floors = np.percentile(luts, alpha_pct, axis=0)
        luts_offset = np.maximum(0, luts - floors)

        ceil = np.percentile(luts_offset, 100 - alpha_pct)
        scale_by = 255. / ceil
        # if only_shift:
        #     scale_by = 1 << int(np.log2(scale_by))
        luts_quantized = np.floor(luts_offset * scale_by).astype(np.int)
        luts_quantized = np.minimum(255, luts_quantized)

        # compute err
        luts_ideal = (luts - luts_offset) * scale_by
        diffs = luts_ideal - luts_quantized
        loss = np.sum(diffs * diffs)

        if loss <= best_loss:
            best_loss = loss
            best_alpha = alpha
            best_floors = floors
            best_scale_by = scale_by

    return best_floors, best_scale_by, best_alpha

# ================================================================ Quantizers

# ------------------------------------------------ Abstract Base Class

class MultiCodebookEncoder(abc.ABC):

    def __init__(self, ncodebooks, ncentroids=256,
                 elemwise_dist_func=dists_elemwise_dot,
                 quantize_lut=False, upcast_every=-1, accumulate_how='sum'):
        self.ncodebooks = ncodebooks
        self.ncentroids = ncentroids
        self.elemwise_dist_func = elemwise_dist_func
        self.quantize_lut = quantize_lut
        self.upcast_every = upcast_every if upcast_every >= 1 else 1
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.upcast_every = min(self.ncodebooks, upcast_every)
        self.accumulate_how = accumulate_how

        self.code_bits = int(np.log2(self.ncentroids))

        # for fast lookups via indexing into flattened array
        self.offsets = (np.arange(self.ncodebooks, dtype=np.int) *
                        self.ncentroids)

    def name(self):
        return "{}_{}x{}b_iters={}_quantize={}".format(
            self.preproc, self.ncodebooks, self.code_bits, self.opt_iters,
            int(self.quantize_lut))

    def params(self):
        return {'_ncodebooks': self.ncodebooks,
                '_code_bits': self.code_bits, 'opt_iters': self.opt_iters,
                '_quantize': self.quantize_lut}

    def _learn_lut_quantization(self, X, Q=None):
        if self.quantize_lut:  # TODO put this logic in separate function
            print("learning quantization...")

            # print("initial Q: ", Q)
            if Q is None:
                # num_rows = min(10 * 1000, len(X) // 2)
                # _, queries = extract_random_rows(
                #     X[num_rows:], how_many=1000, remove_from_X=False)
                # X = X[:num_rows]  # limit to first 10k rows of X
                _, Q = extract_random_rows(
                    X, how_many=1000, remove_from_X=False)
                Q = Q.T  # want each row to be one query, not each col

            # Q = self._pad_ncols(Q)
            # if self.preproc == 'OPQ':
            #     Q = pq.opq_rotate(Q, self.R)
            # elif self.preproc == 'BOPQ':
            #     Q = pq.bopq_rotate(Q, self.rotations)
            # elif self.preproc == 'GEHT':
            #     Q = Q[:, self.perm]

            # print("Q shape: ", Q.shape)

            # compute luts for all the queries
            # luts = [self.encode_Q(q, quantize=False) for q in Q]
            luts = self.encode_Q(Q, quantize=False)
            # luts = np.vstack(luts)
            # print("ncodebooks: ", self.ncodebooks)
            # print("luts shape: ", luts.shape)
            assert luts.shape == (len(Q), self.ncodebooks, self.ncentroids)
            luts = np.moveaxis(luts, 2, 1)
            assert luts.shape == (len(Q), self.ncentroids, self.ncodebooks)
            luts = luts.reshape(len(Q) * self.ncentroids, self.ncodebooks)

            self.lut_offsets, self.scale_by, _ = _learn_best_quantization(luts)
            # print("self.lut_offsets.shape", self.lut_offsets.shape)
            # print("self.scale_by.shape", self.scale_by.shape)
            # print("self.scale_by", self.scale_by)
            assert self.lut_offsets.shape == (self.ncodebooks,)
            # self.lut_offsets = self.lut_offsets[:, np.newaxis]
            self.total_lut_offset = np.sum(self.lut_offsets)
            # print("lut offsets: ", self.lut_offsets)

    def dists_enc(self, X_enc, Q_luts, unquantize=True):
        X_enc = np.ascontiguousarray(X_enc)

        all_dists = np.empty((len(Q_luts), len(X_enc)), dtype=np.float32)
        for i, lut in enumerate(Q_luts):
            centroid_dists = lut.ravel()[X_enc.ravel()]
            dists = centroid_dists.reshape(X_enc.shape)
            if self.upcast_every < 2 or not self.quantize_lut:
                dists = dists.sum(axis=-1)
            else:
                dists = dists.reshape(dists.shape[0], -1, self.upcast_every)
                if self.accumulate_how == 'sum':
                    # sum upcast_every vals, then clip to mirror saturating
                    # unsigned addition, then sum without saturation (like u16)
                    dists = dists.sum(2)
                    dists = np.clip(dists, 0, 255).sum(axis=-1)
                elif self.accumulate_how == 'mean':
                    # mirror hierarchical avg_epu8
                    print("reducing using mean!")
                    while dists.shape[-1] > 2:
                        dists = (dists[:, :, ::2] + dists[:, :, 1::2] + 1) / 2
                    dists = (dists[:, :, 0] + dists[:, :, 1] + 1) / 2
                    dists = dists.sum(axis=-1)  # clipping not needed
                    dists *= self.upcast_every  # convert mean to sum
                else:
                    raise ValueError("accumulate_how must be 'sum' or 'mean'")

            if self.quantize_lut and unquantize:
                dists = (dists / self.scale_by) + self.total_lut_offset
            all_dists[i] = dists

        return all_dists.T


# ------------------------------------------------ Product Quantization

def _learn_centroids(X, ncentroids, ncodebooks, subvect_len):
    ret = np.empty((ncentroids, ncodebooks, subvect_len))
    # print("_learn_centroids(): running kmeans...")
    tot_sse = 0
    X_bar = X - np.mean(X, axis=0)
    col_sses = np.sum(X_bar * X_bar, axis=0) + 1e-14
    tot_sse_using_mean = np.sum(col_sses)

    for i in range(ncodebooks):
        print("running kmeans in subspace {}/{}...".format(
            i + 1, ncodebooks), end=" ")
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        # centroids, labels = kmeans(X_in, ncentroids)
        centroids, labels, sse = kmeans(X_in, ncentroids, return_sse=True)

        # X_bar = X_in - np.mean(X_in, axis=0)
        # sse_using_mean = np.sum(X_bar * X_bar) + 1e-14
        subspace_sse = np.sum(col_sses[start_col:end_col])
        print("mse / {{var(X_subs), var(X)}}: {:.3g}, {:.3g}".format(
            sse / subspace_sse, sse * ncodebooks / tot_sse_using_mean))
        tot_sse += sse
        # print("centroids shape: ", centroids.shape)
        # print("ret shape: ", ret.shape)
        ret[:, i, :] = centroids

    print("--- total mse / var(X): {:.3g}".format(tot_sse / tot_sse_using_mean))

    return ret


def _parse_codebook_params(D, code_bits=-1, bits_per_subvect=-1, ncodebooks=-1):
    if ncodebooks < 0:
        ncodebooks = code_bits // bits_per_subvect
    elif code_bits < 1:
        code_bits = bits_per_subvect * ncodebooks
    elif bits_per_subvect < 1:
        bits_per_subvect = code_bits // ncodebooks

    ncentroids = int(2 ** bits_per_subvect)
    subvect_len = D // ncodebooks

    assert code_bits % bits_per_subvect == 0
    if D % subvect_len:
        print("D, ncodebooks, subvect_len = ", D, ncodebooks, subvect_len)
        assert D % subvect_len == 0  # TODO rm this constraint

    return ncodebooks, ncentroids, subvect_len


def _fit_pq_lut(q, centroids, elemwise_dist_func):
    _, ncodebooks, subvect_len = centroids.shape
    q = q.reshape((1, ncodebooks, subvect_len))
    q_dists = np.sum(centroids * q, axis=-1)

    return q_dists  # ncentroids, ncodebooks, row-major


class PQEncoder(MultiCodebookEncoder):

    def __init__(self, ncodebooks, ncentroids=256,
                 elemwise_dist_func=dists_elemwise_dot,
                 preproc='PQ', encode_algo=None, quantize_lut=False,
                 upcast_every=-1, accumulate_how='sum',
                 **preproc_kwargs):
        super().__init__(
            ncodebooks=ncodebooks, ncentroids=ncentroids,
            elemwise_dist_func=elemwise_dist_func,
            quantize_lut=quantize_lut, upcast_every=upcast_every,
            accumulate_how=accumulate_how)
        self.preproc = preproc
        self.encode_algo = encode_algo
        self.preproc_kwargs = preproc_kwargs

    def _pad_ncols(self, X):
        return ensure_num_cols_multiple_of(X, self.ncodebooks)

    def fit(self, X, Q=None, **preproc_kwargs):
        self.subvect_len = int(np.ceil(X.shape[1] / self.ncodebooks))
        X = self._pad_ncols(X)

        self.centroids = None
        if self.preproc == 'BOPQ':
            self.centroids, _, self.rotations = pq.learn_bopq(
                X, ncodebooks=self.ncodebooks, codebook_bits=self.code_bits,
                **self.preproc_kwargs)
        elif self.preproc == 'OPQ':
            self.centroids, _, self.R = pq.learn_opq(
                X, ncodebooks=self.ncodebooks, codebook_bits=self.code_bits,
                **self.preproc_kwargs)
        elif self.preproc == 'GEHT':
            self.perm = subs.greedy_eigenvector_threshold(
                X, subspace_len=self.subvect_len, **self.preproc_kwargs)
            assert X.shape[1] == len(set(self.perm))
            X = X[:, self.perm]

        if self.centroids is None:
            if self.encode_algo in ('splits', 'multisplits'):
                self.splits_lists, self.centroids = \
                    clusterize.learn_splits_in_subspaces(
                        X, subvect_len=self.subvect_len,
                        nsplits_per_subs=self.code_bits, algo=self.encode_algo)
            else:
                self.centroids = _learn_centroids(
                    X, self.ncentroids, self.ncodebooks, self.subvect_len)

        self._learn_lut_quantization(X, Q)

    def name(self):
        return "{}_{}".format(self.preproc, super().name())

    def params(self):
        d = super().params()
        d['_preproc'] = self.preproc
        return d

    def encode_Q(self, Q, quantize=True):
        # quantize param enables quantization if set in init; separate since
        # quantization learning needs to call this func, but vars like
        # lut_offsets aren't set when this function calls it

        Q = np.atleast_2d(Q)
        Q = self._pad_ncols(Q)
        if self.preproc == 'OPQ':
            Q = pq.opq_rotate(Q, self.R)
        elif self.preproc == 'BOPQ':
            Q = pq.bopq_rotate(Q, self.rotations)
        elif self.preproc == 'GEHT':
            Q = Q[:, self.perm]

        luts = np.zeros((Q.shape[0], self.ncodebooks, self.ncentroids))
        print("Q shape: ", Q.shape)
        for i, q in enumerate(Q):
            lut = _fit_pq_lut(q, centroids=self.centroids,
                              elemwise_dist_func=self.elemwise_dist_func)
            if self.quantize_lut and quantize:
                lut = np.maximum(0, lut - self.lut_offsets)
                lut = np.floor(lut * self.scale_by).astype(np.int)
                lut = np.minimum(lut, 255)
            luts[i] = lut.T
        return luts

    def encode_X(self, X, **sink):
        X = self._pad_ncols(X)
        if self.preproc == 'OPQ':
            X = pq.opq_rotate(X, self.R)
        elif self.preproc == 'BOPQ':
            X = pq.bopq_rotate(X, self.rotations)
        elif self.preproc == 'GEHT':
            X = X[:, self.perm]

        if self.encode_algo in ('splits', 'multisplits'):
            split_type = ('multi' if self.encode_algo == 'multisplits'
                          else 'single')
            idxs = clusterize.encode_using_splits(
                X, self.subvect_len, self.splits_lists, split_type=split_type)
        else:
            idxs = pq._encode_X_pq(X, codebooks=self.centroids)

        return idxs + self.offsets  # offsets let us index into raveled dists
