#!/usr/bin/env python

from __future__ import division, absolute_import

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


# ================================================================ Quantizers

# ------------------------------------------------ Product Quantization

def _learn_centroids(X, ncentroids, nsubvects, subvect_len):
    ret = np.empty((ncentroids, nsubvects, subvect_len))
    # print("_learn_centroids(): running kmeans...")
    tot_sse = 0
    X_bar = X - np.mean(X, axis=0)
    col_sses = np.sum(X_bar * X_bar, axis=0) + 1e-14
    tot_sse_using_mean = np.sum(col_sses)

    for i in range(nsubvects):
        print("running kmeans in subspace {}/{}...".format(
            i + 1, nsubvects), end=" ")
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        # centroids, labels = kmeans(X_in, ncentroids)
        centroids, labels, sse = kmeans(X_in, ncentroids, return_sse=True)

        # X_bar = X_in - np.mean(X_in, axis=0)
        # sse_using_mean = np.sum(X_bar * X_bar) + 1e-14
        subspace_sse = np.sum(col_sses[start_col:end_col])
        print("mse / {{var(X_subs), var(X)}}: {:.3g}, {:.3g}".format(
            sse / subspace_sse, sse * nsubvects / tot_sse_using_mean))
        tot_sse += sse
        # print("centroids shape: ", centroids.shape)
        # print("ret shape: ", ret.shape)
        ret[:, i, :] = centroids

    print("--- total mse / var(X): {:.3g}".format(tot_sse / tot_sse_using_mean))

    return ret


def _parse_codebook_params(D, code_bits=-1, bits_per_subvect=-1, nsubvects=-1):
    if nsubvects < 0:
        nsubvects = code_bits // bits_per_subvect
    elif code_bits < 1:
        code_bits = bits_per_subvect * nsubvects
    elif bits_per_subvect < 1:
        bits_per_subvect = code_bits // nsubvects

    ncentroids = int(2 ** bits_per_subvect)
    subvect_len = D // nsubvects

    assert code_bits % bits_per_subvect == 0
    if D % subvect_len:
        print("D, nsubvects, subvect_len = ", D, nsubvects, subvect_len)
        assert D % subvect_len == 0  # TODO rm this constraint

    return nsubvects, ncentroids, subvect_len


def _fit_pq_lut(q, centroids, elemwise_dist_func):
    _, nsubvects, subvect_len = centroids.shape
    # print("q shape: ", q.shape)
    # print("centroids shape: ", centroids.shape)

    # print("using dist func: ", elemwise_dist_func)

    # q = ensure_num_cols_multiple_of(np.atleast_2d(q), subvect_len)
    # Q = np.atleast_2d(Q)
    # assert len(q) == nsubvects * subvect_len
    q = q.reshape((1, nsubvects, subvect_len))
    # q_dists_ = elemwise_dist_func(centroids, q)
    # q_dists = np.sum(q_dists, axis=-1)
    q_dists = np.sum(centroids * q, axis=-1)

    # return np.asfortranarray(q_dists_)  # ncentroids, nsubvects, col-major
    return q_dists  # ncentroids, nsubvects, row-major


# class PQEncoder(object):

#     def __init__(self, dataset, code_bits=-1, bits_per_subvect=-1,
#                  nsubvects=-1, elemwise_dist_func=dists_elemwise_sq):
#         X = dataset.X_train
#         self.elemwise_dist_func = elemwise_dist_func

#         tmp = _parse_codebook_params(X.shape[1], code_bits=code_bits,
#                                      bits_per_subvect=bits_per_subvect,
#                                      nsubvects=nsubvects)
#         self.nsubvects, self.ncentroids, self.subvect_len = tmp
#         self.code_bits = int(np.log2(self.ncentroids))

#         # for fast lookups via indexing into flattened array
#         self.offsets = np.arange(self.nsubvects, dtype=np.int) * self.ncentroids

#         self.centroids = _learn_centroids(X, self.ncentroids, self.nsubvects,
#                                           self.subvect_len)

#     def name(self):
#         return "PQ_{}x{}b".format(self.nsubvects, self.code_bits)

#     def params(self):
#         return {'_preproc': 'PQ', '_ncodebooks': self.nsubvects,
#                 '_code_bits': self.code_bits}

#     def encode_X(self, X, **sink):
#         idxs = pq._encode_X_pq(X, codebooks=self.centroids)
#         return idxs + self.offsets  # offsets let us index into raveled dists

#     def encode_q(self, q, **sink):
#         return None  # we use fit_query() instead, so fail fast

#     def dists_true(self, X, q):
#         return np.sum(self.elemwise_dist_func(X, q), axis=-1)

#     def fit_query(self, q, **sink):
#         self.q_dists_ = _fit_pq_lut(q, centroids=self.centroids,
#                                     elemwise_dist_func=self.elemwise_dist_func)

#     def dists_enc(self, X_enc, q_unused=None):
#         # this line has each element of X_enc index into the flattened
#         # version of q's distances to the centroids; we had to add
#         # offsets to each col of X_enc above for this to work
#         centroid_dists = self.q_dists_.T.ravel()[X_enc.ravel()]
#         return np.sum(centroid_dists.reshape(X_enc.shape), axis=-1)


def _learn_best_quantization(luts):  # luts can be a bunch of vstacked luts
    best_loss = np.inf
    best_alpha = None
    best_floors = None
    best_scale_by = None
    for alpha in [.001, .002, .005, .01, .02, .05, .1]:
        alpha_pct = int(100 * alpha)

        # compute quantized luts this alpha would yield
        floors = np.percentile(luts, alpha_pct, axis=0)
        luts_offset = np.maximum(0, luts - floors)

        ceil = np.percentile(luts_offset, 100 - alpha_pct)
        scale_by = 255. / ceil
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


class PQEncoder(object):

    def __init__(self, nsubvects, ncentroids=256,
                 elemwise_dist_func=dists_elemwise_dot,
                 preproc='PQ', quantize_lut=False, encode_algo=None,
                 **preproc_kwargs):
        self.nsubvects = nsubvects
        self.ncentroids = ncentroids
        self.elemwise_dist_func = elemwise_dist_func
        self.preproc = preproc
        self.quantize_lut = quantize_lut
        self.encode_algo = encode_algo
        self.preproc_kwargs = preproc_kwargs

        self.code_bits = int(np.log2(self.ncentroids))

        # for fast lookups via indexing into flattened array
        self.offsets = (np.arange(self.nsubvects, dtype=np.int) *
                        self.ncentroids)

    def _pad_ncols(self, X):
        return ensure_num_cols_multiple_of(X, self.nsubvects)

    def fit(self, X, Q=None, **preproc_kwargs):
        self.subvect_len = int(np.ceil(X.shape[1] / self.nsubvects))
        # print("orig X shape: ", X.shape)
        # print("initial X_train shape: ", X.shape)
        # print("ncodebooks:", self.nsubvects)
        X = self._pad_ncols(X)
        # print("X_train shape after padding: ", X.shape)
        # print("nsubvects: ", self.nsubvects)
        # print("subvect_len: ", self.subvect_len)
        # print("------------------------")

        self.centroids = None
        # if self.preproc == 'PQ':
        #     self.centroids = _learn_centroids(
        #         X, self.ncentroids, self.nsubvects, self.subvect_len)
        if self.preproc == 'BOPQ':
            self.centroids, _, self.rotations = pq.learn_bopq(
                X, ncodebooks=self.nsubvects, codebook_bits=self.code_bits,
                **self.preproc_kwargs)
        elif self.preproc == 'OPQ':
            self.centroids, _, self.R = pq.learn_opq(
                X, ncodebooks=self.nsubvects, codebook_bits=self.code_bits,
                **self.preproc_kwargs)
        elif self.preproc == 'GEHT':
            self.perm = subs.greedy_eigenvector_threshold(
                X, subspace_len=self.subvect_len, **self.preproc_kwargs)
            assert X.shape[1] == len(set(self.perm))

            # print("len(self.perm):", len(self.perm))
            # print("len(set(self.perm)):", len(set(self.perm)))
            # import sys; sys.exit()

            X = X[:, self.perm]
            # # just normal PQ after permuting
            # self.centroids = _learn_centroids(
            #     X, self.ncentroids, self.nsubvects, self.subvect_len)
        # else:
        #     raise ValueError("unrecognized preproc: '{}'".format(self.preproc))

        if self.centroids is None:
            if self.encode_algo == 'splits':
                self.splits_lists, self.centroids = \
                    clusterize.learn_splits_in_subspaces(
                        X, subvect_len=self.subvect_len,
                        nsplits_per_subs=self.code_bits)
            else:
                self.centroids = _learn_centroids(
                    X, self.ncentroids, self.nsubvects, self.subvect_len)

        if self.quantize_lut:  # TODO put this logic in separate function
            print("learning quantization...")

            if Q is None:
                # num_rows = min(10 * 1000, len(X) // 2)
                # _, queries = extract_random_rows(
                #     X[num_rows:], how_many=1000, remove_from_X=False)
                # X = X[:num_rows]  # limit to first 10k rows of X
                _, Q = extract_random_rows(
                    X, how_many=1000, remove_from_X=False)

            # compute luts for all the queries
            # luts = [self.encode_Q(q, quantize=False) for q in Q]
            luts = self.encode_Q(Q)
            luts = np.vstack(luts)
            assert luts.shape == (self.ncentroids * len(Q),
                                  self.nsubvects)

            self.lut_offsets, self.scale_by, _ = _learn_best_quantization(luts)
            self.total_lut_offset = np.sum(self.lut_offsets)

    def name(self):
        return "{}_{}x{}b_iters={}_quantize={}".format(
            self.preproc, self.nsubvects, self.code_bits, self.opt_iters,
            int(self.quantize_lut))

    def params(self):
        return {'_preproc': self.preproc, '_ncodebooks': self.nsubvects,
                '_code_bits': self.code_bits, 'opt_iters': self.opt_iters,
                '_quantize': self.quantize_lut}

    def encode_Q(self, Q):
        # was_1d = Q.ndim == 1
        Q = np.atleast_2d(Q)
        # if was_1d:
        #     Q = Q.reshape(1, -1)

        # print("Q shape after made 2d: ", Q.shape)

        # just simplifies impl of _fit_pq_lut
        # Q = ensure_num_cols_multiple_of(Q, self.subvect_len)
        Q = self._pad_ncols(Q)

        # print("Q shape after ensure_num_cols_multiple_of: ", Q.shape)

        if self.preproc == 'OPQ':
            Q = pq.opq_rotate(Q, self.R)
        elif self.preproc == 'BOPQ':
            Q = pq.bopq_rotate(Q, self.rotations)
        elif self.preproc == 'GEHT':
            Q = Q[:, self.perm]

        # print("len(self.perm):", len(self.perm))
        # print("len(set(self.perm)):", len(set(self.perm)))
        # print("Q shape: ", Q.shape)
        # luts = []

        # luts = np.zeros((Q.shape[0], self.ncentroids, self.nsubvects))
        luts = np.zeros((Q.shape[0], self.nsubvects, self.ncentroids))
        # luts = []
        for i, q in enumerate(Q):
            # print("q shape: ", q.shape)
            lut = _fit_pq_lut(q, centroids=self.centroids,
                              elemwise_dist_func=self.elemwise_dist_func)
            if self.quantize_lut:
                assert False # TODO rm after debug
                lut = np.maximum(0, lut - self.lut_offsets)
                lut = np.floor(lut * self.scale_by).astype(np.int)
                lut = np.minimum(lut, 255)
            # luts.append(lut)
            # luts[i] = lut
            luts[i] = lut.T
            # luts.append(lut.T)  # nsubvects x ncentroids

        # if was_1d:
        #     return luts[0]
        return luts
        # luts = [lut[np.newaxis, ...] for lut in luts]
        # return np.stack(luts, axis=0)
        # return np.vstack(luts)

    def encode_X(self, X, **sink):
        # X = ensure_num_cols_multiple_of(X, self.subvect_len)
        X = self._pad_ncols(X)
        if self.preproc == 'OPQ':
            X = pq.opq_rotate(X, self.R)
        elif self.preproc == 'BOPQ':
            X = pq.bopq_rotate(X, self.rotations)
        elif self.preproc == 'GEHT':
            X = X[:, self.perm]

        if self.encode_algo == 'splits':
            idxs = clusterize.encode_using_splits(
                X, self.subvect_len, self.splits_lists)
        else:
            idxs = pq._encode_X_pq(X, codebooks=self.centroids)

        # # TODO rm
        # X_hat = pq.reconstruct_X_pq(idxs, self.centroids)
        # # err = compute_reconstruction_error(X_rotated, X_hat, subvect_len=subvect_len)
        # err = pq.compute_reconstruction_error(X, X_hat)
        # print("X reconstruction err: ", err)

        # import sys; sys.exit()

        return idxs + self.offsets  # offsets let us index into raveled dists

    # def fit_query(self, q, quantize=True, **sink):
    #     quantize = quantize and self.quantize_lut
    #     self.q_dists_ = self.encode_Q(q, quantize=quantize)
    #     if quantize:
    #         # print "min, max lut values: {}, {}".format(np.min(self.q_dists_),
    #         #     np.max(self.q_dists_))
    #         assert np.min(self.q_dists_) >= 0
    #         assert np.max(self.q_dists_) <= 255
    #     if False:
    #         _, axes = plt.subplots(3, figsize=(9, 11))
    #         sb.violinplot(data=self.q_dists_, inner="box", cut=0, ax=axes[0])
    #         axes[0].set_xlabel('Codebook')
    #         axes[0].set_ylabel('Distance to query')
    #         axes[0].set_ylim([0, np.max(self.q_dists_)])

    #         sb.heatmap(data=self.q_dists_, ax=axes[1], cbar=False, vmin=0)
    #         axes[1].set_xlabel('Codebook')
    #         axes[1].set_ylabel('Centroid')

    #         sb.distplot(self.q_dists_.ravel(), hist=False, rug=True,
    #                     vertical=False, ax=axes[2])
    #         axes[2].set_xlabel('Centroid dist to query')
    #         axes[2].set_ylabel('Fraction of centroids')
    #         axes[2].set_xlim([0, np.max(self.q_dists_) + .5])

    #         # plot where the mean is
    #         mean_dist = np.mean(self.q_dists_)
    #         ylim = axes[2].get_ylim()
    #         axes[2].plot([mean_dist, mean_dist], ylim, 'r--')
    #         axes[2].set_ylim(ylim)

    #         plt.show()

    # def dists_enc(self, X_enc, Q_luts, X, Q, unquantize=True): # TODO rm
    def dists_enc(self, X_enc, Q_luts, unquantize=True):
        # this line has each element of X_enc index into the flattened
        # version of q's distances to the centroids; we had to add
        # offsets to each col of X_enc above for this to work
        # was_1d = Q_luts.ndim == 1
        # Q_luts = np.atleast_2d(Q_luts)
        all_dists = np.empty((len(Q_luts), len(X_enc)), dtype=np.float32)
        # print("X_enc shape: ", X_enc.shape)
        # print("Q luts shape: ", Q_luts.shape)
        # # print("len(Q_luts): ", len(Q_luts))

        # print("X_enc col maxes:", np.max(X_enc, axis=0))  # these are right
        # print("X_enc col mins:", np.min(X_enc, axis=0))
        X_enc = np.ascontiguousarray(X_enc)

        # print("filters: ")
        # for q in Q.T:
        #     print(q.reshape(8, 3))

        # X, Q = self._pad_ncols(X), self._pad_ncols(Q.T).T
        # true_prods = X @ Q
        # print("offsets: ", self.offsets)
        # print("X shape, Q shape", X.shape, Q.shape)

        for i, lut in enumerate(Q_luts):
            # print("----- query ", i)
            # print("lut {} has shape: {}".format(i, lut.shape))
            # flat_lut = np.asfortranarray(lut.T).ravel()
            # lut = np.ascontiguousarray(lut)
            centroid_dists = lut.ravel()[X_enc.ravel()]
            # centroid_dists = lut.T.ravel()[X_enc.ravel()]
            # print("centroid dists has shape: ", centroid_dists.shape)
            dists = centroid_dists.reshape(X_enc.shape).sum(axis=-1)

            # # # TODO rm
            # true_dists = true_prods[:, i]
            # diffs = true_dists - dists
            # mse = np.mean(diffs * diffs) / np.var(true_dists)
            # print("mse for query #{}: {:3g}".format(i, mse))
            # print("variance of true dists / 1e3: ", np.var(true_dists) / 1e3)
            # print("variance of dists / 1e3: ", np.var(dists) / 1e3)
            # print("variance of diffs / 1e3: ", np.var(diffs) / 1e3)

            # assert np.sum(np.isnan(dists)) == 0
            # assert np.sum(np.isnan(true_dists)) == 0

            # print("min, max true dist / 1000:",
            #       np.min(true_dists) / 1e3, np.max(true_dists) / 1e3)
            # print("min, max dists hat / 1000:",
            #       np.min(dists) / 1e3, np.max(dists) / 1e3)
            # print("num nans in true_dists, our_dists:",
            #       np.sum(np.isnan(true_dists)), np.sum(np.isnan(dists)))

            # for n in np.random.randint(len(dists), size=20):
            #     print("true dist, dist we computed: {:.1f} vs {:.1f}".format(
            #         true_dists[n] / 1000, dists[n] / 1000))

            # TODO rm
            # alright, let's go subpace by subspace here...
            # if True:
            # if False:
            #     q = Q[:, i]
            #     # print("q: ", q)
            #     # for n in [0]:
            #     for n in np.random.randint(len(dists), size=2):
            #         x = X[n]
            #         x_enc = np.copy(X_enc[n])
            #         x_enc -= self.offsets
            #         # print("x_enc:", x_enc)
            #         true_prod = true_prods[n, i]
            #         prod_hat = 0
            #         true_subs_prods = np.zeros(self.nsubvects)
            #         subs_prods = np.zeros(self.nsubvects)
            #         quantize_errs = np.zeros(self.nsubvects)
            #         for m in range(self.nsubvects):
            #             idx = x_enc[m]

            #             # compute estimated product in this subspace
            #             prod = lut[m, idx]
            #             prod_hat += prod
            #             subs_prods[m] = prod

            #             # compute true product in this subspace
            #             start_idx = m * self.subvect_len
            #             end_idx = start_idx + self.subvect_len
            #             x_subs = x[start_idx:end_idx]
            #             q_subs = q[start_idx:end_idx]
            #             true_subs_prods[m] = np.sum(x_subs * q_subs)

            #             # compute quantization err
            #             centroid = self.centroids[idx, m]
            #             diffs = x_subs - centroid
            #             quantize_errs[m] = np.sum(diffs * diffs)
            #         # print("true prod, prod_hat", true_prod / 1000, prod_hat / 1000)
            #         # print("sum true prods, sum prods hat:",
            #         #       true_subs_prods.sum() / 1000, subs_prods.sum() / 1000)
            #         print("true dist, dist we computed: ",
            #               true_dists[n] / 1000, dists[n] / 1000)
            #         print("true_subs_prods: ", true_subs_prods / 1000)
            #         print("subs_prods_hat:  ", subs_prods / 1000)
            #         # print("quantize errs:", quantize_errs / 1000)

            if self.quantize_lut and unquantize:
                assert False  # make sure not accidentally quantizing for now
                dists = (dists / self.scale_by) + self.total_lut_offset
            all_dists[i] = dists

        # print("computed dists; exiting")
        # import sys; sys.exit()

        # return all_dists.ravel() if was_1d else all_dists.T
        return all_dists.T
