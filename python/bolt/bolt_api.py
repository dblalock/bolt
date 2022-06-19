#!/usr/bin/env python

# TODO maybe have sklearn transforms for dot prod and Lp dists
# TODO add L1 distance

from . import bolt  # inner bolt because of SWIG

import kmc2  # state-of-the-art kmeans initialization (as of NIPS 2016)
import numpy as np
from sklearn import cluster, exceptions


# ================================================================ Distances

def dists_elemwise_sq(x, q):
    diffs = x - q
    return diffs * diffs


def dists_elemwise_l1(x, q):
    return np.abs(x - q)


def dists_elemwise_dot(x, q):
    return x * q


# ================================================================ Preproc

def _insert_zeros(X, nzeros):
    """injects nzeros zero columns spaced as far apart as possible"""
    if nzeros < 1:
        return X

    N, D = X.shape
    D_new = D + nzeros
    X_new = np.zeros((N, D_new), dtype=X.dtype)

    nonzeros_per_zero = D // nzeros
    if nonzeros_per_zero < 1:
        X_new[:, :D] = X
        return X_new

    stripe_width = nonzeros_per_zero
    for i in range(nzeros):
        in_start = stripe_width * i
        in_end = in_start + stripe_width
        out_start = i * (stripe_width + 1)
        out_end = out_start + stripe_width
        X_new[:, out_start:out_end] = X[:, in_start:in_end]
    out_end += 1

    remaining_len = D - in_end
    out_remaining_len = D_new - out_end
    # print "D, remaining_incols, remaining_outcols, in_end, out_end: ", \
    #     D, remaining_len, out_remaining_len, in_end, out_end
    assert remaining_len == out_remaining_len
    assert remaining_len >= 0
    if remaining_len:
        X_new[:, out_end:out_end+remaining_len] = X[:, in_end:D]

    # check that we copied both the beginning and end properly
    # assert np.array_equal(X[:, 0], X_new[:, 1])
    assert np.array_equal(X[:, 0], X_new[:, 0])
    if remaining_len > 0:
        assert np.array_equal(X[:, -1], X_new[:, -1])

    return X_new


def _ensure_num_cols_multiple_of(X, multiple_of):
    """Adds as many columns of zeros as necessary to ensure that
    X.shape[1] % multiple_of == 0"""
    remainder = X.shape[1] % multiple_of
    if remainder > 0:
        return _insert_zeros(X, multiple_of - remainder)
    return X


# ================================================================ kmeans

def kmeans(X, k, max_iter=16, init='kmc2'):
    X = X.astype(np.float32)
    np.random.seed(123)

    # if k is huge, initialize centers with cartesian product of centroids
    # in two subspaces
    if init == 'subspaces':
        sqrt_k = int(np.sqrt(k) + .5)
        if sqrt_k ** 2 != k:
            raise ValueError("K must be a square number if init='subspaces'")

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

    estimator = cluster.MiniBatchKMeans(k, init=seeds, max_iter=max_iter).fit(X)
    return estimator.cluster_centers_, estimator.labels_


# ================================================================ PQ

# TODO rm after debug
def _encode_X_pq(X, codebooks, elemwise_dist_func=dists_elemwise_sq):
    ncentroids, ncodebooks, subvect_len = codebooks.shape

    assert X.shape[1] == (ncodebooks * subvect_len)

    idxs = np.empty((X.shape[0], ncodebooks), dtype=np.int)
    X = X.reshape((X.shape[0], ncodebooks, subvect_len))
    for i, row in enumerate(X):
        row = row.reshape((1, ncodebooks, subvect_len))
        dists = elemwise_dist_func(codebooks, row)
        dists = np.sum(dists, axis=2)
        idxs[i, :] = np.argmin(dists, axis=0)

    # return idxs + self._offsets_  # offsets let us index into raveled dists
    return idxs  # [N x ncodebooks]


def _learn_centroids(X, ncentroids, ncodebooks):
    subvect_len = int(X.shape[1] / ncodebooks)
    assert subvect_len * ncodebooks == X.shape[1]  # must divide evenly
    ret = np.empty((ncentroids, ncodebooks, subvect_len))
    for i in range(ncodebooks):
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        centroids, labels = kmeans(X_in, ncentroids)
        ret[:, i, :] = centroids

    return ret.astype(np.float32)


def _learn_best_quantization(luts):  # luts can be a bunch of vstacked luts
    best_loss = np.inf
    best_alpha = None
    best_floors = None
    best_scale_by = None
    for alpha in [0, .001, .002, .005, .01, .02, .05, .1]:
        alpha_pct = int(100 * alpha)
        # compute quantized luts this alpha would yield
        floors = np.percentile(luts, alpha_pct, axis=0)
        luts_offset = np.maximum(0, luts - floors)  # clip at 0

        ceil = np.percentile(luts_offset, 100 - alpha_pct)
        scale_by = 255. / ceil
        luts_quantized = np.floor(luts_offset * scale_by).astype(np.int)
        luts_quantized = np.minimum(255, luts_quantized)  # clip at 255

        # compute err
        luts_ideal = (luts - luts_offset) * scale_by
        diffs = luts_ideal - luts_quantized
        loss = np.sum(diffs * diffs)

        # print "alpha = {}\t-> loss = {}".format(alpha, loss)
        # # yep, almost exactly alpha saturate in either direction
        # print "fraction of 0s, 255s = {}, {}".format(
        #     np.mean(luts_offset == 0), np.mean(luts_quantized == 255))

        if loss <= best_loss:
            best_loss = loss
            best_alpha = alpha
            best_floors = floors
            best_scale_by = scale_by

    # print "best alpha, loss = ", best_alpha, best_loss
    # print "best floors, scale = ", best_floors, best_scale_by

    return best_floors, best_scale_by, best_alpha


def _extract_random_rows(X, how_many, remove_from_X=True):
    if how_many > len(X):
        raise IndexError("how_many ({}) > len(X) ({})".format(how_many, len(X)))
    split_start = np.random.randint(len(X) - how_many - 1)
    split_end = split_start + how_many
    rows = np.copy(X[split_start:split_end])
    if remove_from_X:
        return np.vstack((X[:split_start], X[split_end:])), rows
    return X, rows


def _fit_pq_lut(q, centroids, elemwise_dist_func):
    _, nsubvects, subvect_len = centroids.shape
    assert len(q) == nsubvects * subvect_len

    q = q.reshape((1, nsubvects, subvect_len))
    q_dists_ = elemwise_dist_func(centroids, q)
    q_dists_ = np.sum(q_dists_, axis=-1)

    return np.asfortranarray(q_dists_)  # ncentroids, nsubvects, col-major


def _learn_quantization_params(X, centroids, elemwise_dist_func, Q=None,
                               # plot=True):
                               plot=False):
    """learn distros of entries in each lut"""

    if Q is None:
        num_rows = int(min(10*1000, len(X) / 2))
        how_many = int(min(1000, num_rows // 2))
        _, Q = _extract_random_rows(
            X[num_rows:], how_many=how_many, remove_from_X=False)
        X = X[:num_rows]  # limit to first 10k rows of X

    # compute luts for all the queries
    luts = [_fit_pq_lut(q, centroids=centroids,
                        elemwise_dist_func=elemwise_dist_func) for q in Q]
    luts = np.vstack(luts)

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sb
        # print "plotting LUT distributions..."

        plot_luts = np.asfortranarray(luts[:5000])
        _, ax = plt.subplots(figsize=(10, 4))
        sb.violinplot(data=plot_luts, inner="box", cut=0, ax=ax)
        ax.set_title('Distributions of distances within each LUT')
        ax.set_xlabel('LUT')
        ax.set_ylabel('Distance to query')
        ax.set_ylim([0, np.max(plot_luts)])

        plt.show()

    # print "lut stats (min, mean, max):"
    # print np.min(luts, axis=0)
    # print np.mean(luts, axis=0)
    # print np.max(luts, axis=0)

    assert luts.shape == (centroids.shape[0] * len(Q), centroids.shape[1])

    offsets, scaleby, _ = _learn_best_quantization(luts)
    return offsets.astype(np.float32), scaleby


class MockEncoder(object):
    """Stand-in for cpp impl; only for debuging"""

    def __init__(self, nbytes):
        self._enc_bytes = nbytes
        self.ncodebooks = 2 * nbytes
        self._encoder = bolt.BoltEncoder(nbytes)

    def set_centroids(self, centroids):
        # accept centroids as 2D array like cpp; but we'll need them 3D
        nrows, ndims = centroids.shape
        ncentroids = 16
        codebook_sz = ncentroids * ndims

        self.centroids = np.empty((ncentroids, self.ncodebooks, ndims))
        for m in range(self.ncodebooks):
            start_idx = m * ncentroids  # start idx of block
            end_idx = start_idx + ncentroids
            block = centroids[start_idx:end_idx]
            self.centroids[:, m, :] = block

        # check whether centroids bridge is broken
        self._encoder.set_centroids(centroids)
        raw_centroids = self._encoder.centroids()
        cpp_centroids = np.full(raw_centroids.shape, -1)

        # print "ncentroids, ncodebooks, ndims ", self.centroids.shape

        inbuff = raw_centroids.ravel()
        outbuff = np.zeros(raw_centroids.size) - 1
        for m in range(self.ncodebooks):
            start_idx = m * codebook_sz  # start idx of block
            for i in range(ncentroids):  # for each row in block
                for j in range(ndims):  # for each col in block
                    in_idx = start_idx + (ndims * i) + j
                    out_idx = start_idx + (ncentroids * j) + i
                    outbuff[in_idx] = inbuff[out_idx]

        cpp_centroids = outbuff.reshape(centroids.shape)

        # print "py, cpp centroids: "
        # print centroids[:20]
        # print cpp_centroids[:20]
        # print centroids.shape
        # print cpp_centroids.shape

        assert np.allclose(centroids, cpp_centroids)

    def set_data(self, X):
        self.X = X
        self.X_enc = _encode_X_pq(X, self.centroids)
        ncodebooks = self.centroids.shape[1]
        enc_offsets = np.arange(ncodebooks, dtype=np.int) * 16

        self._encoder.set_data(X)
        raw_Xenc = self._encoder.codes()
        assert 2 * raw_Xenc.shape[1] == ncodebooks
        cpp_Xenc = np.empty((raw_Xenc.shape[0], ncodebooks), dtype=np.uint8)
        # cpp returns codes in bitpacked form, so unpack them
        for in_j, out_j in enumerate(range(0, ncodebooks, 2)):
            col = raw_Xenc[:, in_j]
            cpp_Xenc[:, out_j] = np.bitwise_and(col, 15)
        for in_j, out_j in enumerate(range(1, ncodebooks, 2)):
            col = raw_Xenc[:, in_j]
            cpp_Xenc[:, out_j] = np.bitwise_and(col, 255 - 15) >> 4

        # print "python X enc"
        # print self.X_enc.shape
        # print self.X_enc[:20]
        # print "cpp X enc"
        # print cpp_Xenc.shape
        # print cpp_Xenc[:20]
        # print "raw cpp X_enc"
        # print raw_Xenc[:20]

        self.X_enc += enc_offsets

    def set_offsets(self, offsets):
        assert self.scale > 0
        self._offsets_ = offsets
        self._encoder.set_offsets(offsets)

    def set_scale(self, scale):
        self.scale = scale
        self._encoder.set_scale(scale)

    def _quantize_lut(self, raw_lut):
        lut = np.floor(raw_lut * self.scale + self._offsets_)
        return np.maximum(0, np.minimum(lut, 255)).astype(np.uint16)

    def _dists(self, raw_lut):
        lut = np.asfortranarray(self._quantize_lut(raw_lut))
        centroid_dists = lut.T.ravel()[self.X_enc.ravel()]
        return np.sum(centroid_dists.reshape(self.X_enc.shape), axis=-1)
        # dists = np.sum(centroid_dists.reshape(self.X_enc.shape), axis=-1)

    def dists_sq(self, q):
        lut = _fit_pq_lut(q, centroids=self.centroids,
                          elemwise_dist_func=dists_elemwise_sq)

        offsets_cpp = self._encoder.get_offsets()
        scale_cpp = self._encoder.get_scale()

        # print "py, cpp offsets:"
        # print self.offsets
        # print offsets_cpp

        # print "py, cpp scale factors:"
        # print self.scale
        # print scale_cpp

        lut_py = self._quantize_lut(lut)
        # print "lets try to read the cpp lut..."
        # self._encoder.lut_l2(q)
        self._encoder.lut_dot(q)
        lut_cpp = self._encoder.get_lut()

        # print "py, cpp lut:"  # within +/- 1 using naive lut impl in cpp
        # print lut_py
        # print lut_cpp

        # return self._dists(lut)
        dists_py = self._dists(lut)
        dists_cpp = self._encoder.dists_sq(q)[:len(dists_py)]  # strip padding

        # print "py, cpp initial dists:"
        # print dists_py[:20]
        # print dists_cpp[:20]

        # print "py, cpp final dists:"
        # print dists_py[-20:]
        # print dists_cpp[-20:]

        return dists_py
        # return dists_cpp

    def dot_prods(self, q):
        lut = _fit_pq_lut(q, centroids=self.centroids,
                          elemwise_dist_func=dists_elemwise_dot)
        return self._dists(lut)


class Reductions:
    SQUARED_EUCLIDEAN = 'l2'
    DOT_PRODUCT = 'dot'


class Accuracy:
    LOWEST = 'lowest'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


_acc_to_nbytes = {
    Accuracy.LOWEST: 2,
    Accuracy.LOW: 8,
    Accuracy.MEDIUM: 16,
    Accuracy.HIGH: 32,
}


class Encoder(object):

    def __init__(self, reduction=Reductions.SQUARED_EUCLIDEAN,
                 accuracy=Accuracy.MEDIUM, norm_mean=None):
        self._enc_bytes = _acc_to_nbytes[accuracy]
        self.reduction = reduction
        self.norm_mean = norm_mean if norm_mean is not None \
            else reduction != Reductions.DOT_PRODUCT

    def _preproc(self, X):
        # TODO rows of X also needs to have variance >> 1 to avoid
        # everything going to 0 when bolt_encode converts to ints in argmin
        one_d = len(X.shape) == 1
        if one_d:
            X = X.reshape((1, -1))
        ncodebooks = self._enc_bytes * 2
        X = X.astype(np.float32)
        if self.norm_mean:
            # X = X - self.means_
            X -= self.means_
        out = _ensure_num_cols_multiple_of(X.astype(np.float32), ncodebooks)
        return out.ravel() if one_d else out

    @property
    def nbytes(self):
        try:
            return self._nbytes_
        except AttributeError:
            raise exceptions.NotFittedError("Encoder has not yet been given "
                                            "a dataset; call fit() first")

    def fit(self, X, just_train=False, Q=None):
        if not len(X.shape) == 2:
            raise IndexError("X must be [num_examples x num_dimensions]!")
        if X.shape[1] < 2 * self._enc_bytes:
            raise ValueError("num_dimensions must be at least 2 * nbytes")

        ncentroids = 16
        self._nbytes_ = self._enc_bytes * len(X)  #

        self.DEBUG = False
        # self.DEBUG = True

        self.means_ = np.mean(X, axis=0) if self.norm_mean \
            else np.zeros(X.shape[1])
        self.means_ = self.means_.astype(np.float32)
        # self.means_ = np.zeros_like(self.means_) # TODO rm
        # self.means_ = np.ones_like(self.means_) # TODO rm

        X = self._preproc(X)
        self._ndims_ = X.shape[1]
        self._ncodebooks = self._enc_bytes * 2

        centroids = _learn_centroids(X, ncentroids=ncentroids,
                                     ncodebooks=self._ncodebooks)
        centroids = centroids.astype(np.float32)

        # print "X shape, centroids shape: ", X.shape, centroids.shape
        # print "X means before preproc:", self.means_
        # print "X means after preproc:", np.mean(X, axis=0)
        # print "means of centroids:", np.mean(centroids, axis=0)

        if self.DEBUG:
            self._encoder_ = MockEncoder(self._enc_bytes)
        else:
            self._encoder_ = bolt.BoltEncoder(self._enc_bytes)

        # print "centroids shape: ", centroids.shape

        # compute lut offsets and scaleby for l2 and dot here; we'll have
        # to switch off which ones are used based on which method gets called
        if self.reduction == Reductions.SQUARED_EUCLIDEAN:
            elemwise_dist_func = dists_elemwise_sq
        elif self.reduction == Reductions.DOT_PRODUCT:
            elemwise_dist_func = dists_elemwise_dot
        else:
            self._bad_reduction()

        offsets, self.scale = _learn_quantization_params(
            X, centroids, elemwise_dist_func)
        # account for fact that cpp's fma applies scale first, then adds offset
        # self._offsets_ = -offsets / self.scale
        self._offsets_ = -offsets * self.scale
        self._total_offset_ = np.sum(self._offsets_)

        # offsets_sq, self.scale_sq_ = _learn_quantization_params(
            # X, centroids, dists_elemwise_sq)
        # offsets_dot, self.scale_dot_ = _learn_quantization_params(
            # X, centroids, dists_elemwise_dot)

        self._encoder_.set_scale(self.scale)
        self._encoder_.set_offsets(self._offsets_)

        # # account for fact that cpp applies scale first, then offset, in fma
        # self.offsets_sq_ = -offsets_sq / self.scale_sq_
        # self.offsets_dot_ = -offsets_dot / self.scale_dot_

        # # TODO rm after debug
        # self.offsets_sq_ *= 5
        # self.offsets_sq_[:] = 0.
        # self.offsets_dot_[:] = 0.
        # self.scale_sq_ = 1.
        # self.scale_dot_ = 1.

        # print "centroids shape", centroids.shape

        # munge centroids into contiguous 2D array;
        # starts as [ncentroids, ncodebooks, subvect_len] and
        # needs to be [ncentroids * ncodebooks, subvect_len
        subvect_len = centroids.shape[-1]
        flat_centroids = np.empty((self._ncodebooks * ncentroids,
                                   subvect_len), dtype=np.float32)
        for m in range(self._ncodebooks):
            codebook = centroids[:, m, :]
            start_row = m * ncentroids
            end_row = start_row + ncentroids
            flat_centroids[start_row:end_row, :] = codebook

        # print "centroids shape: ", centroids.shape
        # print "flat centroids shape: ", flat_centroids.shape

        self._encoder_.set_centroids(flat_centroids)

        if not just_train:
            self._encoder_.set_data(X)
            self._n = len(X)

        return self

    def set_data(self, X):
        """set data to actually encode; separate from fit() because fit()
        could use different training data than what we actully compress"""
        self._encoder_.set_data(self._preproc(X))
        self._n = len(X)

    def transform(self, q, unquantize=False):
        if self.reduction == Reductions.DOT_PRODUCT:
            func = self._encoder_.dot_prods
        elif self.reduction == Reductions.SQUARED_EUCLIDEAN:
            func = self._encoder_.dists_sq
        else:
            self._bad_reduction()

        ret = func(self._preproc(q))[:self._n]
        return (ret - self._total_offset_) / self.scale if unquantize else ret

    def knn(self, q, k):
        if self.reduction == Reductions.DOT_PRODUCT:
            return self._encoder_.knn_mips(self._preproc(q), k)
        elif self.reduction == Reductions.SQUARED_EUCLIDEAN:
            return self._encoder_.knn_l2(self._preproc(q), k)

    def _bad_reduction(self):
        raise ValueError("Unreconized reduction '{}'!".format(self.reduction))


    # def dot(self, q, unquantize=False):
    #     self._check_reduction(Reductions.DOT_PRODUCT)
    #     ret = self._encoder_.dot_prods(self._preproc(q))[:self._n]
    #     return (ret - self._offsets_) * self.scale if unquantize else ret

    # def dists_sq(self, q, unquantize=False):
    #     self._check_reduction(Reductions.SQUARED_EUCLIDEAN)
    #     ret = self._encoder_.dists_sq(self._preproc(q))[:self._n]
    #     return (ret - self._offsets_) * self.scale if unquantize else ret

    # def knn_dot(self, q, k):
    #     self._check_reduction(Reductions.DOT_PRODUCT)
    #     return self._encoder_.knn_mips(self._preproc(q), k)

    # def knn_l2(self, q, k):
    #     self._check_reduction(Reductions.SQUARED_EUCLIDEAN)
    #     return self._encoder_.knn_l2(self._preproc(q), k)


def _test_insert_zeros():
    X = np.random.randn(4, 1000)
    for ncols in range(1, X.shape[1] + 1):
        for nzeros in np.arange(64):
            _insert_zeros(X[:, :ncols], nzeros)


if __name__ == '__main__':
    _test_insert_zeros()
