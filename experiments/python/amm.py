#!/bin/env/python

import abc
import numpy as np
# from sklearn.decomposition import PCA, SparsePCA
from sklearn import decomposition
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA
from sklearn.utils.extmath import randomized_svd
import numba  # conda install numba

# import ffht  # https://github.com/FALCONN-LIB/FFHT; python setup.py install
import scipy

from joblib import Memory
_memory = Memory('.', verbose=1, compress=9)

KEY_NMULTIPLIES = 'muls'

OSNAP_DEFAULT_S = 4
# OSNAP_DEFAULT_S = 2


# ================================================================ utils

def _nmultiplies_matmul(A, B):
    return A.shape[0] * A.shape[1] * B.shape[1]


def _nmultiplies_matmul_with_sizes(N, D, M):
    return N * D * M


def _nmultiplies_svd(N, D):
    return min(N * N * D, N * D * D)


def _nmultiplies_qr(N, D):
    return min(N * N * D, N * D * D)


# ================================================================ types

class InvalidParametersException(Exception):
    pass


class ApproxMatmul(abc.ABC):

    def __init__(*args_unused, **kwargs_unused):
        pass

    def fit(self, A, B, Y=None):  # Y = A @ B if not specified
        pass

    def set_A(self, A):
        pass

    def set_B(self, B):
        pass

    def reset_for_new_task(self):
        pass

    @abc.abstractmethod
    def __call__(self, A, B):
        pass

    def predict(self, A, B):
        return self(A, B)

    def get_params(self):
        return {}

    # def get_nmuls(self, A, B, fixedA=False, fixedB=False):
    @abc.abstractmethod
    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        pass


class ExactMatMul(ApproxMatmul):

    def __call__(self, A, B):
        return A @ B

    def get_speed_metrics(self, A, B, **sink):
        return {KEY_NMULTIPLIES: _nmultiplies_matmul(A, B)}


def _scalar_quantize(A, axis=1, signed=False, nbits=8):
    unsigned_maxval = float(1 << int(nbits)) - 1

    # # TODO rm
    # # return np.zeros((A.shape[0], 1)), np.ones((A.shape[0], 1)), A
    # # offsets = np.zeros((A.shape[0], 1))
    # offsets = A.min(axis=1, keepdims=True)
    # # scales = maxval / np.ones((A.shape[0], 1))
    # scales = maxval / A.max(axis=1, keepdims=True)
    # Aq = (A - offsets) * scales
    # return offsets, scales, Aq

    # maxval = float(1 << int(nbits)) - 1
    mins = A.min(axis=axis, keepdims=True)
    # A_offset = A - offsets
    ranges = (A - mins).max(axis=axis, keepdims=True) + 1e-20
    scales = unsigned_maxval / ranges
    # Aq = (A_offset * (maxval / scales)).astype(np.int)
    # Aq = (A_offset * scales).astype(np.int)

    if signed:
        # sign_offset = 1 << (nbits - 1)  # 8 bits -> 128
        # A_offset -= sign_offset
        offsets = mins + (ranges * (128. / 255))
        minval = -(1 << (nbits - 1))
        maxval = -minval - 1
    else:
        offsets = mins
        minval = 0
        maxval = (1 << nbits) - 1

    Aq = (A - offsets) * scales
    # print("min, max A:", Aq.min(), Aq.max())  # looks good
    Aq = np.clip(Aq, minval, maxval).astype(np.int)

    return offsets, scales, Aq


class QuantizedMatmul(ApproxMatmul):
    __slots__ = 'nbits a_offsets a_scales b_offsets b_scales A B'.split()

    def __init__(self, nbits=8):
        self.nbits = nbits

    def __call__(self, A, B):
        assert A.shape[1] == B.shape[0]  # dims need to match
        N, D = A.shape
        D, M = B.shape
        if self.A is None:
            self.set_A(A)
        if self.B is None:
            self.set_B(B)

        # print("QuantizedMatmul")
        # print("min, max A:", self.A.min(), self.A.max())
        # print("min, max A offsets:", self.a_offsets.min(), self.a_offsets.max())
        # print("min, max A scales    :", self.a_scales.min(), self.a_scales.max())
        # print("min, max B:", self.B.min(), self.B.max())
        # print("min, max B offsets:", self.b_offsets.min(), self.b_offsets.max())
        # print("min, max B scales    :", self.b_scales.min(), self.b_scales.max())

        # ((A - a_offsets) / a_scales) @ ((B - b_offsets) / b_scales)  # noqa
        # ignoring scales, we have:
        # (A - a_off) @ (B - b_off)
        # = A @ B - (a_off @ B) - (A @ b_off) + a_off @ b_off
        # maxval = (1 << int(self.nbits)) - 1
        ret = (self.A @ self.B).astype(np.float32)
        ret *= 1. / self.a_scales
        ret *= 1. / self.b_scales

        A_off = np.tile(self.a_offsets, (1, D))
        B_off = np.tile(self.b_offsets, (D, 1))

        return ret + (A_off @ B) + (A @ B_off) - (A_off @ B_off)

    def set_A(self, A):
        # unsigned quantization; we *could* learn the offsets and scales
        # on the training set, but since this is a baseline, we're giving it
        # the advantage of using the "true" offsets/scales
        self.a_offsets, self.a_scales, self.A = _scalar_quantize(
            A, axis=1, signed=False, nbits=self.nbits)

        # mins = A.min(axis=1, keepdims=True)
        # A_offset = A - mins
        # scales = A_offset.max(axis=1, keepdims=True) + 1e-20
        # self.A = (A_offset * (255. / scales)).astype(np.int)

    def set_B(self, B):
        # signed quantization (for maddubs instruction)
        self.b_offsets, self.b_scales, self.B = _scalar_quantize(
            B, axis=0, signed=True, nbits=self.nbits)
        # self.b_offsets, self.b_scales, self.B = _scalar_quantize(
        #     B.T, nbits=self.nbits, signed=True)
        # # quantize each col, not each row
        # self.b_offsets = self.b_offsets.ravel()
        # self.b_scales = self.b_scales.ravel()
        # self.B = self.B.T

    def reset_for_new_task(self):
        self.A = None
        self.B = None

    def get_speed_metrics(self, A, B, **sink):
        # neglect packing, postprocessing, etc
        return {KEY_NMULTIPLIES: _nmultiplies_matmul(A, B)}


class SketchedMatmul(ApproxMatmul, abc.ABC):
    __slots__ = 'd'

    def __init__(self, d):
        self.d = int(d)

    def get_params(self):
        return {'d': self.d}

    def sketch(self, A, B):
        pass

    def call(self, A, B):
        A_hat, B_hat = self.sketch(A, B)
        assert A_hat.shape[0] == A.shape[0]
        assert B_hat.shape[1] == B.shape[1]
        assert A_hat.shape[1] <= self.d  # verify sketch size not cheating
        return A_hat @ B_hat

    def __call__(self, A, B):
        assert A.shape[1] == B.shape[0]  # dims need to match
        D = A.shape[1]
        if D <= self.d:
            raise InvalidParametersException(
                'D <= d: {} < {}'.format(D, self.d))
        if B.shape[1] <= self.d:
            raise InvalidParametersException(
                'M <= d: {} < {}'.format(B.shape[1], self.d))
        return self.call(np.copy(A), np.copy(B))  # guarantee A, B unchanged

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        assert not (fixedA and fixedB)  # this would be stupid, so fail fast
        sketch_nmuls = self._get_nmuls(A.shape[0], A.shape[1], B.shape[1],
                                       self.d, fixedA=fixedA, fixedB=fixedB)
        N, D = A.shape
        D, M = B.shape
        sketched_matmul_nmuls = N * self.d * M
        return {KEY_NMULTIPLIES: sketch_nmuls + sketched_matmul_nmuls}

    def _get_nmuls(self, N, D, M, d, fixedA=False, fixedB=False):
        # default nmuls = sketching with dense matrix
        nmuls = 0
        if not fixedA:
            nmuls += N * D * d
        if not fixedB:
            nmuls += M * D * d
        return nmuls


class RandGaussSketch(SketchedMatmul):

    def sketch(self, A, B):
        D = A.shape[1]
        V = np.random.randn(D, self.d).astype(np.float32)
        # dividing by expected norm is more similar to theory papers,
        # but no reason this should actually be better AFAIK
        # V /= np.sqrt(D)
        V /= np.linalg.norm(V, axis=0)
        A = A @ V
        B = V.T @ B
        return A, B


class RandOrthoGaussSketch(SketchedMatmul):

    def sketch(self, A, B):
        D = A.shape[1]
        V = np.random.randn(D, self.d).astype(np.float32)
        V, _ = np.linalg.qr(V)
        A = A @ V
        B = V.T @ B
        return A, B


class RandRademacherSketch(SketchedMatmul):

    def sketch(self, A, B):
        D = A.shape[1]
        V = np.random.randint(2, size=(D, self.d)).astype(np.float32) * 2 - 1
        V /= np.sqrt(D)
        A = A @ V
        B = V.T @ B
        return A, B


class HadamardSketch(SketchedMatmul):

    def sketch(self, A, B):
        D = A.shape[1]
        use_D = 1 << int(np.ceil(np.log2(D)))
        V = scipy.linalg.hadamard(use_D)[:D, :self.d].astype(np.float32)
        V /= np.linalg.norm(V, axis=0)
        # V /= np.sqrt(2)
        # V *= np.sqrt(2)
        # V *= np.sqrt(D / self.d)
        # V *= (D / self.d) ** .25
        A = A @ V
        B = V.T @ B
        return A, B


class SketchSqSample(SketchedMatmul):

    def sketch(self, A, B):
        return sketch_sq_sample(A, B, self.d)

    def _get_nmuls(self, N, D, M, d, **sink):
        return _nmultiplies_sketch_sq_sample(N, D, M, d)


class FdAmm(SketchedMatmul):

    def sketch(self, A, B):
        return fd_amm_sketches(A, B, self.d)

    def _get_nmuls(self, N, D, M, d, **sink):
        return _nmultiplies_fd_amm_sketches(N, D, M, d)


class CooccurSketch(SketchedMatmul):

    def sketch(self, A, B):
        return cooccur_sketches(A, B, self.d)

    def _get_nmuls(self, N, D, M, d, **sink):
        return _nmultiplies_cooccur_sketches(N, D, M, d)


class FastJlSketch(SketchedMatmul):

    def sketch(self, A, B):
        return fastjl_sketches(A, B, self.d)

    def _get_nmuls(self, N, D, M, d, **sink):
        return _nmultiplies_fastjl_sketches(N, D, M, d)


class HashJlSketch(SketchedMatmul):

    def sketch(self, A, B):
        return hash_sketches(A, B, self.d)

    def _get_nmuls(self, N, D, M, d, **sink):
        return _nmultiplies_hash_sketches(N, D, M, d)


class OsnapSketch(SketchedMatmul):

    def sketch(self, A, B):
        return osnap_sketches(A, B, self.d, s=OSNAP_DEFAULT_S)

    # def get_params(self):
    #     return {'d': self.d, 's': OSNAP_DEFAULT_S}

    def _get_nmuls(self, N, D, M, d, **sink):
        return _nmultiplies_osnap_sketches(N, D, M, d)


class SvdSketch(SketchedMatmul):
    __slots__ = 'd niters Ua SVTa Ub SVTb'.split()

    def __init__(self, d, niters=5):
        self.d = d
        self.niters = niters
        self.reset_for_new_task()

    def get_params(self):
        return {'d': self.d, 'niters': self.niters}

    def _check_mat_shape(self, M):
        if M is None:
            return False
        # if np.min(M.shape) < self.d:
        if np.max(M.shape) < self.d:
            raise InvalidParametersException(
                'shape has entry < d: {} < {}'.format(M.shape, self.d))
        return True

    def set_A(self, A):
        # if A is None:
        #     return
        if self._check_mat_shape(A):
            self.Ua, self. SVTa = svd_sketch(A, self.d)

    def set_B(self, B):
        if self._check_mat_shape(B):
            self.Ub, self.SVTb = svd_sketch(B, self.d)

    def reset_for_new_task(self):
        self.Ua = None
        self.SVTa = None
        self.Ub = None
        self.SVTb = None

    # def __call__(self, A=None, B=None):
    #     assert A.shape[1] == B.shape[0]  # dims need to match
    #     if A.shape[1] < self.d:
    #         raise InvalidParametersException('D < d')

    def call(self, A=None, B=None):
        if self.Ua is None:
            self.set_A(A)
        if self.Ub is None:
            self.set_B(B)
        D = self.Ua.shape[1]
        if D < self.d:
            raise InvalidParametersException(
                'D < d: {} < {}'.format(D, self.d))
        # verify sketch size isn't cheating
        # print("A.shape", A.shape)
        # print("B.shape", B.shape)
        # print("self.Ua.shape: ", self.Ua.shape)
        # print("self.SVTa.shape: ", self.SVTa.shape)
        # print("self.Ub.shape: ", self.Ub.shape)
        # print("self.SVTb.shape: ", self.SVTb.shape)
        # print("self.d: ", self.d)
        assert self.Ua.shape[1] <= self.d
        assert self.SVTa.shape[0] <= self.d
        assert self.SVTb.shape[0] <= self.d
        assert self.Ub.shape[1] <= self.d
        # innermost parens important so that matmuls actually use low rank
        # outer parens help if B ncols < A nrows (which is true for us)
        return self.Ua @ ((self.SVTa @ self.Ub) @ self.SVTb)

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        # XXX this will break if not called right after self.call()
        total = 0
        d = self.d
        N, D = A.shape
        _, M = B.shape
        if not fixedA:
            total += _nmultiplies_svd_sketch(N, D, d, niters=self.niters)
        if not fixedB:
            total += _nmultiplies_svd_sketch(D, M, d, niters=self.niters)
        total += d * D * d  # SVTa @ UB, d x D @ D x d
        total += d * d * M  # (above) @ SVTb, d x d @ d x M
        total += N * d * M  # Ua @ (above), N x d @ d x M
        return {KEY_NMULTIPLIES: total}


@_memory.cache
def _fitted_pca(X, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit(X)


class TrainedPcaSketch(ApproxMatmul):
    __slots__ = 'pca d A B V'.split()

    def __init__(self, d):
        # self.pca = PCA(n_components=d)
        self.d = d
        self.reset_for_new_task()

    def reset_for_new_task(self):
        self.A = None
        self.B = None

    def fit(self, A, B, Y=None):  # Y = A @ B if not specified
        D, M = B.shape
        print("called fit on TrainedPcaSketch!")
        if D < self.d:
            raise InvalidParametersException(
                'D < d: {} < {}'.format(D, self.d))
        if M < self.d:
            raise InvalidParametersException(
                'M < d: {} < {}'.format(M, self.d))

        self.pca = _fitted_pca(A, n_components=self.d)
        self.V = self.pca.components_.T
        # print("components V.T @ V =\n", self.V.T @ self.V) # yep, orthonormal

    def set_A(self, A):
        self.A = A @ self.V

    def set_B(self, B):
        self.B = self.V.T @ B

    def __call__(self, A, B):
        assert A.shape[1] == B.shape[0]  # dims need to match
        if B.shape[1] < self.d:
            raise InvalidParametersException(
                'M < d: {} < {}'.format(B.shape[1], self.d))

        if (self.A is None):
            self.set_A(A)
        if (self.B is None):
            self.set_B(B)
        return self.A @ self.B

    def get_params(self):
        return {'d': self.d}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        d = self.d
        nmuls = N * d * M  # assuming matrices already sketched
        if not fixedA:
            nmuls += N * D * d
        if not fixedB:
            nmuls += D * M * d
        return {KEY_NMULTIPLIES: nmuls}


@_memory.cache
def _fitted_sparse_pca(X, d, unscaled_alpha, **kwargs):
    # this seems to work better than initializing with MiniBatchSparsePCA,
    # svd of cov mat, or basically anything else I tried
    U, _, Vt = randomized_svd(X, n_components=d, random_state=123)
    U = U[:, :d]
    V = Vt.T[:d]

    # SparsePCA (and all the sklearn dictionary learning stuff)
    # internally uses sum of squared errs for each sample, and L1 norm
    # of parameter matrix; to make alpha meaningful across datasets,
    # want to scale by number of examples (so it's effectively using MSE)
    # and divide by L1 norm (which grows linearly with size of parameter
    # matrix / vector); also scale by variance of data for similar reasons
    N, D = X.shape
    alpha = unscaled_alpha * np.var(X - X.mean(axis=0)) * N / D
    verbose = 1
    pca = SparsePCA(n_components=d, alpha=alpha, normalize_components=True,
                    method='lars', U_init=U, V_init=V, max_iter=10,
                    ridge_alpha=max(1, len(X) * X.std() * 10),
                    # ridge_alpha=1e8,
                    verbose=verbose, random_state=123)
    if verbose > 0:
        print("fitting sparse pca...")
    return pca.fit(X)


class TrainedSparsePcaSketch(ApproxMatmul):
    __slots__ = 'pca d alpha nnz can_optimize_transform A B'.split()

    # def __init__(self, d, alpha, can_optimize_transform=True):
    def __init__(self, d, alpha, can_optimize_transform=False):
        self.d = d
        self.alpha = alpha
        self.can_optimize_transform = can_optimize_transform
        self.reset_for_new_task()

    def reset_for_new_task(self):
        self.A = None
        self.B = None

    def fit(self, A, B, Y=None):  # Y = A @ B if not specified
        D, M = B.shape
        # if M <= self.d:
        #     raise InvalidParametersException(
        #         'M <= d: {} < {}'.format(M, self.d))
        if D <= self.d:
            raise InvalidParametersException(
                'D < d: {} < {}'.format(D, self.d))

        self.pca = _fitted_sparse_pca(A, d=self.d, unscaled_alpha=self.alpha)
        self.nnz = np.sum(self.pca.components_ != 0)
        sparsity = np.mean(self.pca.components_ == 0)

        if self.nnz < self.d:
            raise InvalidParametersException(
                "ignoring SparsePCA with nnz < d: "
                "{} < {}".format(self.nnz, self.d))
        if sparsity == 0.:
            raise InvalidParametersException(
                "ignoring SparsePCA with no zeros")

    def set_A(self, A):
        if self.can_optimize_transform:
            # uses ridge regression to get coeffs, instead of linear projection
            # disabled by default because it produces garbage on caltech and
            # is more expensive than just doing the matmul
            self.A = self.pca.transform(A)
            self.A += self.pca.mean_ @ self.pca.components_.T
        else:
            self.A = A @ self.pca.components_.T

    def set_B(self, B):
        if self.can_optimize_transform:
            self.B = self.pca.transform(B.T).T
            self.B += (self.pca.mean_ @ self.pca.components_.T).reshape(-1, 1)
        else:
            self.B = (B.T @ self.pca.components_.T).T

    def __call__(self, A, B):
        assert A.shape[1] == B.shape[0]  # dims need to match
        N, D = A.shape
        D, M = B.shape

        if D <= self.d:
            raise InvalidParametersException(
                'D < d: {} < {}'.format(D, self.d))

        fixedA = self.A is not None
        fixedB = self.B is not None

        nmuls_naive = N * D * M
        nmuls_ours = self.get_speed_metrics(
            A, B, fixedA=fixedA, fixedB=fixedB)[KEY_NMULTIPLIES]
        if nmuls_naive <= nmuls_ours:
            raise InvalidParametersException(
                "naive # of multiplies < sparse sketch # of multiplies: "
                "{} < {}".format(nmuls_naive, nmuls_ours))

        if not fixedA:
            self.set_A(A)
        if not fixedB:
            self.set_B(B)

        # if N == 700:
        # if False:
            print("got to weird dset!")
            # print("pca means: ", self.pca.mean_[::20])
            # print("A means:", A.mean(axis=0)[::20])
            # print("B means:", B.mean(axis=1)[::20])
            print("pca means sum: ", self.pca.mean_.sum())
            print("A means sum: ", A.mean(axis=0).sum())
            print("B means sum: ", B.mean(axis=1).sum())
            offsets = (self.pca.mean_ @ self.pca.components_.T)
            print("offsets: ", offsets)
            print("offsets sum: ", offsets.sum())
            # C = (A @ B)
            # print("true mean of output: ", C.mean())
            # print("true std of output: ", C.std())

        return self.A @ self.B

    def get_params(self):
        return {'d': self.d, 'alpha': self.alpha,
                'canCheat': self.can_optimize_transform}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        nmuls_sketch_X = N * self.nnz
        nmuls_sketch_W = M * self.nnz
        nmuls_make_output = N * self.d * M
        total_nmuls = nmuls_make_output
        if not fixedA:
            total_nmuls += nmuls_sketch_X
        if not fixedB:
            total_nmuls += nmuls_sketch_W

        try:  # compute degree of sparsity
            nnz = self.nnz
            sparsity = (self.pca.components_ == 0).mean()
        except AttributeError:  # model not fitted yet
            nnz = -1
            sparsity = -1

        return {KEY_NMULTIPLIES: total_nmuls,
                'nnz': nnz, 'sparsity': sparsity}


# ================================================================ drineas06

def _compute_dim_scores(A, B, A_col_norms=None, B_row_norms=None):
    if A_col_norms is None:
        A_col_norms = np.linalg.norm(A, axis=0)
    if B_row_norms is None:
        B_row_norms = np.linalg.norm(B, axis=1)
    return A_col_norms * B_row_norms


def sketch_sq_sample(A, B, d):
    scores = _compute_dim_scores(A, B)
    idxs, weights = importance_sample(scores, d)
    # idxs, weights = sample_varopt_1d(scores, d)  # doesn't help
    return A[:, idxs] / weights, B[idxs]
    # weights = np.sqrt(weights)
    # return A[:, idxs] / weights, B[idxs] / weights.reshape(-1, 1)

    # probs = scores / np.sum(scores)

    # D = A.shape[1]
    # keep_idxs = np.random.choice(D, size=d, p=probs)
    # # keep_idxs = np.random.choice(D, size=d, p=probs, replace=False)
    # # keep_idxs = np.random.choice(D, size=d, replace=False)
    # # keep_idxs = np.arange(D-1)
    # # keep_idxs = np.arange(1, D)
    # # keep_idxs = np.arange(D)

    # weights = np.sqrt(d * probs)  # what the paper says; huge errors
    # # weights = np.sqrt(D * probs)  # slightly less bad
    # # weights = np.sqrt(np.sqrt(d * probs))
    # # weights = np.ones(D)
    # A = np.copy(A) / weights
    # B = np.copy(B) / weights.reshape(-1, 1)

    # return np.copy(A[:, keep_idxs]), np.copy(B[keep_idxs])
    # return A[:, keep_idxs], B[keep_idxs]
    # return A, B


def _nmultiplies_sketch_sq_sample(N, D, M, d):
    scores_nmuls = N * D + M * D  # sum of sizes of each mat
    reweight_nmuls = N * d + M * d  # sum of sizes of each sampled mat
    return scores_nmuls + reweight_nmuls  # neglect normalization of probs, etc


def sketch_sq_deterministic(A, B, d):
    scores = _compute_dim_scores(A, B)
    D = A.shape[1]
    keep_idxs = np.argsort(scores)[::-d]

    weights = np.sqrt(d * (1. / D))  # uniform prob
    return A[:, keep_idxs] / weights, B[keep_idxs] / weights.reshape(-1, 1)


def test_sketch_sq_sample():
    print("test_sketch_sq_sample")
    N, M, D = 100, 50, 200
    np.random.seed(1234)
    # A = np.random.randint(5, size=(N, D)).astype(np.float32)
    # B = np.random.randint(5, size=(D, M)).astype(np.float32)
    # A -= np.mean(A)
    # B -= np.mean(B)
    A = np.random.randn(N, D).astype(np.float32)
    B = np.random.randn(D, M).astype(np.float32)

    AB = A @ B
    orig_frob_sq = np.mean(AB * AB)
    print("true mss: ", orig_frob_sq)

    prev_normed_err = np.inf
    for d in (10, 20, 30, 40, 50):
        A_hat, B_hat = sketch_sq_sample(A, B, d)
        # A_hat, B_hat = sketch_sq_deterministic(A, B, d)
        AB_hat = A_hat @ B_hat
        # print("AB_hat mss: ", (AB_hat * AB_hat).mean())
        diffs = AB - AB_hat
        err_frob_sq = np.mean(diffs * diffs)
        normed_err_sq = err_frob_sq / orig_frob_sq
        # print("orig mss: ", orig_frob_sq)
        print('d = {}, err = {:.3f}'.format(d, normed_err_sq))
        assert normed_err_sq < 2.
        assert normed_err_sq < (prev_normed_err + .05)  # should usually hold
        prev_normed_err = normed_err_sq


# ================================================================ sampling

# wait, this just returns points summing to the true sample sum
# deterministically...
def importance_sample(sample_weights, m, replace=False):
    probs = sample_weights / sample_weights.sum()
    idxs = np.random.choice(
        np.arange(len(sample_weights)), p=probs, replace=replace, size=m)
    weights = 1. / (probs[idxs] * m)
    return idxs, weights


def _invert_permutation(permutation):
    return np.arange(len(permutation))[np.argsort(permutation)]


def _sum_for_tau(x, tau):
    above_tau = x > tau
    return x[above_tau].sum() + (x[~above_tau] / tau).sum()


def _compute_new_tau(x_sorted_desc, m, tau=0):
    x = x_sorted_desc
    current_sum = _sum_for_tau(x, tau)
    assert current_sum >= m
    while current_sum > m:
        x = x[:-1]
        current_sum = _sum_for_tau(x, tau)


def sample_varopt_1d(x, m):
    # varopt sampling; original paper (see Algorithm 1 on p16):
    #   https://arxiv.org/pdf/0803.0473.pdf
    # better intuition:
    #   https://datasketches.github.io/docs/Sampling/VarOptSampling.html
    #
    # unlike paper, we're just going to do it all at once since that will
    # be simpler and vectorize way better; basically just recursively
    # take largest point w_i if w_i > (m / sum_i w_i), with m decremented
    # by 1 each time; if this doesn't take all the points, importance sample
    # from the remaining points (with probs proportional to their weights)
    #
    # EDIT: this sucks unless really heavy tailed, so probably not a
    # correct impl?
    x = np.asarray(x, dtype=np.float32)
    n = len(x)

    if m >= n:
        return np.arange(n)

    maxval = np.max(x)
    minval = np.min(x)
    assert minval >= 0  # needs nonnegative entries
    if minval == maxval or m == 1:
        return np.random.choice(np.arange(n), size=m)

    sort_idxs = np.argsort(x)[::-1]  # in descending order
    x_sorted = x[sort_idxs]
    unsort_idxs = _invert_permutation(sort_idxs)

    q = x_sorted * (m / np.sum(x_sorted))  # sums to m

    # q_tailsums = np.cumsum(q[::-1])[::-1]

    # next_val = x_sorted[0]
    head_sz = 0
    for i in range(m):
        if q[0] >= 1.:
            head_sz += 1
            q = q[1:] * ((m - 1) / q[1:].sum())
            # TODO just compute tail sums once for renormalization (below)
            # q_mass_eliminated = q[i]
            # next_val = q[i + 1] * (m - head_sz) / m * ()
            # renormalize such that tail sums to m - 1
        else:
            break

    tail_sz = m - head_sz
    # print("m, head_sz, tail_sz:", m, head_sz, tail_sz)
    # print("len(q)", len(q))

    # probs = q / np.sum(q)
    probs = x_sorted[head_sz:] / np.sum(x_sorted[head_sz:])
    tail_idxs = np.random.choice(
        np.arange(head_sz, n), p=probs, replace=False, size=tail_sz)
    idxs = list(tail_idxs)
    # idxs = tail_idxs
    # tau = tail_sz / np.sum(x_sorted[head_sz:])
    # print("tau: ", tau)
    # print("x_sorted[:head_sz + 1]: ", x_sorted[:head_sz + 1])
    # tau = x_sorted[head_sz]
    true_probs = probs[tail_idxs - head_sz] * (tail_sz / m)
    weights = list(1. / (m * true_probs))  # small err; definitely right
    # weights = [tau] * tail_sz

    if head_sz > 0:
        head_idxs = list(np.arange(head_sz))
        head_weights = list(np.ones(head_sz))
        idxs = head_idxs + idxs
        weights = head_weights + weights

    return unsort_idxs[idxs], np.array(weights)


# ============================================================ random sketches

# sketch both A and B jointly using the same matrix to amortize overhead and
# because it seems like this should help accuracy
# @numba.jit(nopython=True)
def fastjl_sketches(A, B, d, P=None):
    N, D = A.shape
    M = B.shape[1]

    # pad A and B for FHT
    log2_D = int(np.ceil(np.log2(D)))
    D_pad = 2 ** log2_D
    A_pad = np.zeros((N, D_pad), dtype=np.float32)
    A_pad[:, :D] = A
    B_pad = np.zeros((D_pad, M), dtype=np.float32)
    B_pad[:D] = B

    # construct and apply random signs for each dim
    randsigns = np.random.randint(0, 2, size=D_pad) * 2 - 1
    # scale now instead of scaling FHT mat, so only O(D) multiplies
    randsigns = randsigns.astype(np.float32) * (1. / np.sqrt(D_pad))
    A_pad *= randsigns
    B_pad *= randsigns.reshape(-1, 1)

    # # apply fast hadamard transform
    H = scipy.linalg.hadamard(D_pad, dtype=np.float32)
    # H = scipy.linalg.hadamard(D_pad, dtype=np.float32) / np.sqrt(D_pad)
    A_pad = A_pad @ H
    B_pad = H @ B_pad

    # dimensionalty reduction
    if P is None:
        # logd = np.log2(D_pad)
        keep_prob = log2_D * log2_D / D_pad
        # if (keep_prob) >= 1:
        # print("WARNING: FastJL returning all zeros mat...")
        P = (np.random.uniform(size=(D_pad, d)) > keep_prob).astype(np.float32)
        # P *= np.random.randn(*P.shape) * (d / keep_prob)
        # scaling sigma totally fails; need norm to actually be 1, not just
        # have expected value of 1
        P *= np.random.randn(*P.shape)
        P *= (1. / np.linalg.norm(P, axis=0))

    # print("P shape, Apad shape, Bpad shape: ", P.shape, A_pad.shape, B_pad.shape)
    return A_pad @ P, P.T @ B_pad


def _nmultiplies_fastjl_sketches(N, D, M, d):  # avg, not exact, since P sparse
    # technically adds or subs, but you'd do fma ops regardless for floats
    log2_D = int(np.ceil(np.log2(D)))
    D_pad = 2 ** log2_D

    fht_nmuls = D_pad * np.log2(D_pad)
    sign_nmuls = D_pad

    # trickier part; expected number of madds (or similar ops) to mul by P
    construct_P_nmuls = D_pad * d  # assuming only 1 mul for rng + threshold
    keep_prob = log2_D * log2_D / D_pad
    nnz_p = min(1, keep_prob) * D_pad  # expected nnz per row of P
    p_nmuls = N * nnz_p * d + d * nnz_p * M

    return fht_nmuls + sign_nmuls + construct_P_nmuls + p_nmuls


@numba.jit(nopython=True)
def hash_sketches(A, B, d, scale=1., share_projections=True):
    N, D = A.shape
    D, M = B.shape
    A_hat = np.zeros((N, d), dtype=A.dtype)
    B_hat = np.zeros((d, M), dtype=B.dtype)

    for j in range(D):
        idx = np.random.randint(d)
        sign = (np.random.randint(0, 2) * 2) - 1
        # coeff = sign * scale  # worse than predicting mean, esp for small d
        coeff = sign * scale / np.sqrt(2)  # actually pretty decent
        # coeff = sign * scale * ((d / D) ** .25)
        # coeff = sign * scale * np.sqrt(d / D)  # best for small d / D
        # coeff = sign * scale * d / D  # best for larger d / D
        A_hat[:, idx] += A[:, j] * coeff
        if share_projections:
            B_hat[idx] += B[j] * coeff
            continue

        # use a different projection for B
        idx = np.random.randint(d)
        sign = (np.random.randint(0, 2) * 2) - 1
        B_hat[idx] += B[j] * sign

    # using unscaled signs preserves norms really well, at least for
    # random matrices
    # print("A norm, A_hat norm:", np.linalg.norm(A), np.linalg.norm(A_hat))
    # print("B norm, B_hat norm:", np.linalg.norm(B), np.linalg.norm(B_hat))
    # A_norm = np.linalg.norm(A)
    # B_norm = np.linalg.norm(B)
    # A_hat *= np.linalg.norm(A) / np.linalg.norm(A_hat)
    # B_hat *= np.linalg.norm(B) / np.linalg.norm(B_hat)

    return A_hat, B_hat


def osnap_sketches(A, B, d, s=OSNAP_DEFAULT_S):
    N, D = A.shape
    D, M = B.shape
    s = max(1, min(d // 2, s))  # handle s too large relative to d
    A_hat = np.zeros((N, d), dtype=A.dtype)
    B_hat = np.zeros((d, M), dtype=B.dtype)

    scale = 1. / np.sqrt(s)
    # scale = 1. / s
    # scale = 1  # seems to often work better than dividing by 1/sqrt(s)?
    # scale = np.sqrt(s)
    # scale = s

    subspace_len = (d + s - 1) // s  # round up
    for ss in range(s):
        start_idx = ss * subspace_len
        end_idx = min(D, start_idx + subspace_len)
        A_hat[:, start_idx:end_idx], B_hat[start_idx:end_idx] = \
            hash_sketches(A, B, subspace_len, scale=scale)

    # A_hat /= np.linalg.norm(A_hat, axis=)

    return A_hat, B_hat


def _nmultiplies_hash_sketches(N, D, M, d):
    # technically adds or subs, but you'd do fma ops regardless for floats
    return N * D + D * M


def _nmultiplies_osnap_sketches(N, D, M, d, s=4):
    return 4 * _nmultiplies_hash_sketches(N, D, M, d)


def test_rand_sketches():
    print("test_svd_sketches")
    N, M, D = 100, 80, 50
    np.random.seed(1234)
    A = np.random.randint(5, size=(N, D)).astype(np.float32)
    B = np.random.randint(5, size=(D, M)).astype(np.float32)
    A -= np.mean(A)
    B -= np.mean(B)

    AB = A @ B
    orig_frob_sq = np.sum(AB * AB)

    prev_normed_err = np.inf
    # for d in [10]:
    for d in (1, 2, 4, 8, 16, 32):
        # (Ua, SVTa), (Ub, SVTb) = svd_sketches(A, B, d)
        # AB_hat = Ua @ (SVTa @ Ub) @ SVTb
        A_hat, B_hat = fastjl_sketches(A, B, d)
        # A_hat, B_hat = hash_sketches(A, B, d)  # sharing projections helps
        # A_hat, B_hat = hash_sketches(A, B, d, share_projections=False)
        # A_hat, B_hat = osnap_sketches(A, B, d)
        AB_hat = A_hat @ B_hat

        # print("fused mats shapes: ")
        # print(Ua.shape, SVTa.shape, Ub.shape, SVTb.shape)

        diffs = AB - AB_hat
        err_frob_sq = np.sum(diffs * diffs)
        normed_err_sq = err_frob_sq / orig_frob_sq
        print('d = {}, err = {:.5f}'.format(d, normed_err_sq))
        # assert normed_err_sq < 1.
        # assert normed_err_sq < prev_normed_err + .001
        prev_normed_err = normed_err_sq


# ================================================================ Rand SVD

def svd_sketch(A, d, niters=5, **kwargs):
    # assert A.shape[0] >= d
    # assert A.shape[1] >= d
    assert np.max(A.shape) >= d  # can't truncate to larger size
    U, S, Vt = randomized_svd(A, n_components=d, n_iter=niters, **kwargs)
    # print("Vt shape: ", Vt.shape)
    # print("S: ", S)
    return (U, np.diag(S) @ Vt)


def _nmultiplies_svd_sketch(N, D, d, niters):
    # # "In contrast, randomized schemes can produce an approximate SVD using
    # # only O(mn log(k) + (m + n)k2) flops" -Halko et al. 2010
    # # https://arxiv.org/pdf/0909.4061.pdf
    # iter_cost = N * D * int(np.ceil(np.log2(d)))
    # iter_cost += (N + D) * d * d
    # return iter_cost * niters

    # # assumes algorithm 4.4 in above; sklearn randomized_svd source
    # # code says it implements algorithm 4.3, but paper says 4.3 should actually
    # # be implemented as 4.4 in practice. Also 4x4's complexity is much easier
    # # to understand and counting multiplies is at best a rough estimate
    # # regardless.
    # #
    # # shapes:
    # #   A:              N x D
    # #   A*:             D x N
    # #   Omega:          D x d
    # #   Y0 = A @ Omega: N x d
    # #   Q0:             N x d
    # #   R0:             d x d
    # #   Y_tilde_j:
    # # gauss_mat_cost = D * d
    # # Y0_cost = N * D * d
    # Y0_cost = N * D * int(np.ceil(np.log2(d)))  # subsampled FFT; see text
    # Y0_cost += _nmultiplies_qr(N, d)
    # Yj_tilde_cost = D * N * d + _nmultiplies_qr(N, d)
    # Yj_cost =

    # okay, sklearn says it uses algorithm 4.3 in Halko et al. 2010 [1],
    # so we're going to go with that
    # [1] https://arxiv.org/pdf/0909.4061.pdf
    # shapes:
    #   A:              N x D
    #   A.T:            D x N
    #   G (Omega):      D x d
    #   A @ G:          N x d
    #   A.T @ (AG)      D x d
    #   A @ (A.T@A@G)   N x d
    #   Q0:             N x d
    #   R0:             d x d
    Omega_cost = D * d
    A_Omega_cost = N * D * d
    # each iter: premul by A.T, then A; assumes no LU or QR for stability
    iter_cost = D * N * d + N * D * d
    return Omega_cost + A_Omega_cost + iter_cost * niters


def svd_sketches(A, B, d, **kwargs):
    return svd_sketch(A, d, **kwargs), svd_sketch(B, d, **kwargs)
    # Ua, Sa, VTa = randomized_svd(A, n_components=d, **kwargs)
    # Ub, Sb, VTb = randomized_svd(B, n_components=d, **kwargs)

    # print("truncated svd mat shapes:")
    # print(Ua.shape, Sa.shape, VTa.shape)
    # print(Ub.shape, Sb.shape, VTb.shape)

    # return (Ua, np.diag(Sa) @ VTa), (Ub, np.diag(Sb) @ VTb)


def test_svd_sketches():
    print("test_svd_sketches")
    N, M, D = 100, 80, 50
    np.random.seed(1234)
    A = np.random.randint(5, size=(N, D)).astype(np.float32)
    B = np.random.randint(5, size=(D, M)).astype(np.float32)
    A -= np.mean(A)
    B -= np.mean(B)

    AB = A @ B
    orig_frob_sq = np.sum(AB * AB)

    prev_normed_err = np.inf
    # for d in [10]:
    for d in (1, 2, 4, 8, 16, 32):
        (Ua, SVTa), (Ub, SVTb) = svd_sketches(A, B, d)
        AB_hat = Ua @ (SVTa @ Ub) @ SVTb

        # print("fused mats shapes: ")
        # print(Ua.shape, SVTa.shape, Ub.shape, SVTb.shape)

        diffs = AB - AB_hat
        err_frob_sq = np.sum(diffs * diffs)
        normed_err_sq = err_frob_sq / orig_frob_sq
        print('d = {}, err = {:.5f}'.format(d, normed_err_sq))
        assert normed_err_sq < 1.
        assert normed_err_sq < prev_normed_err
        prev_normed_err = normed_err_sq


# ================================================================ FD methods

# TODO impl fast-FD, which zeros out half the entries
def frequent_directions(A, d, variant=None):
    N, D = A.shape
    H = np.zeros((d, D))

    assert N >= d
    assert D >= d

    # for i in range(N):
    H[:d - 1] = A[:d - 1]
    for i in range(d - 1, N):
        H[-1] = A[i]
        try:
            U, S, Vt = np.linalg.svd(H, full_matrices=False)  # d x d, d, d x D
        except np.linalg.LinAlgError as e:
            print("SVD failed at iter ", i - (d - 1))
            print("H shape: ", H.shape)
            print("A shape: ", A.shape)
            print("d: ", d)
            # print("svd mat shape: ", U.shape, S.shape, Vt.shape)
            raise e
        # cutoff = S[d - 1]  # S is returned as a vector, not a diagonal mat
        if variant == 'robust':
            raise NotImplementedError()
        else:
            S = np.sqrt((S - S[-1]) ** 2)  # note that last entry is dth entry
            # print("new S shape: ", S.shape)
        # H = np.diag(S) @ Vt  # d x D
        H = Vt * S.reshape(-1, 1)  # d x D; equivalent to np.diag(S) @ Vt

    return H


def fast_frequent_directions(A, d, variant=None, alpha=.5):
    N, D = A.shape
    # H = np.zeros((d, D))
    H = np.copy(A[:d])

    assert N >= d
    assert D >= d

    cutoff_idx = int(d * (1 - alpha))
    cutoff_idx = min(d - 1, cutoff_idx)  # always zero at least last element
    ntrailing_zeros = d - cutoff_idx

    i = d
    while i < N:
        try:
            U, S, Vt = np.linalg.svd(H, full_matrices=False)  # d x d, d, d x D
        except np.linalg.LinAlgError as e:
            print("SVD failed at iter ", i - (d - 1))
            print("H shape: ", H.shape)
            print("A shape: ", A.shape)
            print("d: ", d)
            # print("svd mat shape: ", U.shape, S.shape, Vt.shape)
            raise e

        cutoff = S[cutoff_idx]
        if variant == 'parametrized':
            raise NotImplementedError()
        else:
            S = np.sqrt(np.maximum(S - cutoff, 0) ** 2)
            S = np.sqrt((S - S[-1]) ** 2)  # note that last entry is dth entry
            # print("new S shape: ", S.shape)
        # H = np.diag(S) @ Vt  # d x D
        H = Vt * S.reshape(-1, 1)  # d x D; equivalent to np.diag(S) @ Vt

        # replace zeroed-out rows of H with next rows of A
        end_dim = min(N, i + ntrailing_zeros)
        nrows_to_copy = end_dim - i
        end_row = cutoff_idx + nrows_to_copy
        assert nrows_to_copy <= ntrailing_zeros
        assert end_row <= d
        H[-nrows_to_copy:] = A[i:end_dim]
        i = end_dim

    return H


def parametrized_fd_sketches(A, B, d):
    # from "Improved Practical Matrix Sketching with Guarantees"
    A_hat = fast_frequent_directions(A.T, d, variant='parametrized', alpha=.2)
    B_hat = fast_frequent_directions(B.T, d, variant='parametrized', alpha=.2)
    return A_hat.T, B_hat.T


def fd_amm_sketches(A, B, d):
    # print("A shape: ", A.shape)
    # print("B shape: ", B.shape)
    G = np.hstack((A.T, B))   # D x (N + M)
    H = frequent_directions(G, d)
    assert H.shape == (d, A.shape[0] + B.shape[1])
    C = H[:, :A.shape[0]]  # d x N
    D = H[:, A.shape[0]:]  # d x M
    return C.T, D


def fast_fd_amm_sketches(A, B, d):
    # print("A shape: ", A.shape)
    # print("B shape: ", B.shape)
    G = np.hstack((A.T, B))   # D x (N + M)
    H = fast_frequent_directions(G, d)
    assert H.shape == (d, A.shape[0] + B.shape[1])
    C = H[:, :A.shape[0]]  # d x N
    D = H[:, A.shape[0]:]  # d x M
    return C.T, D


def _nmultiplies_frequent_directions(N, D, d):
    niters = N - d + 1
    iter_svd_cost = _nmultiplies_svd(d, D)
    iter_reweight_cost = d * D
    iter_cost = iter_svd_cost + iter_reweight_cost
    return niters * iter_cost


def _nmultiplies_fast_frequent_directions(N, D, d):
    niters = int(np.ceil(N / d))
    iter_svd_cost = _nmultiplies_svd(d, D)
    iter_reweight_cost = d * D
    iter_cost = iter_svd_cost + iter_reweight_cost
    return niters * iter_cost


def _nmultiplies_fd_amm_sketches(N, D, M, d):
    N, D = D, N + M  # matrices get concatenated
    return _nmultiplies_frequent_directions(N, D, d)


def test_fd_amm_sketches():
    print("test_fd_amm_sketches")
    N, M, D = 100, 80, 50
    np.random.seed(1234)
    A = np.random.randint(5, size=(N, D)).astype(np.float32)
    B = np.random.randint(5, size=(D, M)).astype(np.float32)
    # A -= np.mean(A)
    # B -= np.mean(B)

    AB = A @ B
    orig_frob_sq = np.sum(AB * AB)

    prev_normed_err = np.inf
    for d in (1, 2, 4, 8, 16, 32):
        A_hat, B_hat = fd_amm_sketches(A, B, d)
        AB_hat = A_hat @ B_hat

        diffs = AB - AB_hat
        err_frob_sq = np.sum(diffs * diffs)
        normed_err_sq = err_frob_sq / orig_frob_sq
        print('d = {}, err = {:.5f}'.format(d, normed_err_sq))
        assert normed_err_sq < 1.05
        assert normed_err_sq < prev_normed_err
        prev_normed_err = normed_err_sq


# ================================================================ Co-occurring

def cooccur_sketches(A, B, d):
    N, D = A.shape
    B = B.T
    M, _ = B.shape
    assert B.shape[1] == D

    # assert N >= d  # not enough rows in specified A matrix
    # assert M >= d  # not enough cols in specified B matrix

    # add new rows to A or B so that R from QR factorization is at least d x d
    if N < d:
        A_new = np.zeros((d, D), dtype=A.dtype)
        A_new[:N] = A
        A = A_new
    if M < d:
        B_new = np.zeros((d, D), dtype=B.dtype)
        B_new[:M] = B
        B = B_new

    X = np.copy(A[:, :d])   # N x d
    Y = np.copy(B[:, :d])    # M x d

    # mid_idx = d - 2  # does this make it work better for large d? EDIT: nope
    mid_idx = d // 2
    ntrailing_zeros = d - mid_idx

    i = d
    while i < D:
        Qx, Rx = np.linalg.qr(X)  # N x d, d x d
        Qy, Ry = np.linalg.qr(Y)  # M x d, d x d
        prod = Rx @ Ry.T          # d x d
        U, S, Vt = np.linalg.svd(prod, full_matrices=False)  # d x d, d, d x d
        cutoff = S[mid_idx]
        S = np.sqrt(np.maximum(S - cutoff, 0))

        # print("prod.shape", prod.shape)
        # print("orig X.shape", X.shape)
        # print("orig Y.shape", Y.shape)

        X = Qx @ (U * S)  # equivalent to U @ np.diag(S)
        Y = Qy @ (Vt.T * S)  # equivalent to Vt.T @ np.diag(S)

        # print("X.shape", X.shape)
        # print("Qx.shape", Qx.shape)
        # print("U.shape", U.shape)

        # replace zeroed-out cols of X and Y with new cols of A and B
        end_dim = min(D, i + ntrailing_zeros)
        ncols_to_copy = end_dim - i
        end_col = mid_idx + ncols_to_copy
        assert ncols_to_copy <= ntrailing_zeros
        assert end_col <= d

        X[:, mid_idx:end_col] = A[:, i:end_dim]
        Y[:, mid_idx:end_col] = B[:, i:end_dim]
        i = end_dim

    return X[:N], Y[:M].T  # slicing is because we may have zero-padded


def _nmultiplies_cooccur_sketches(N, D, M, d):
    niters = int(np.ceil(D / d))
    iter_qr_cost = _nmultiplies_qr(N, d) + _nmultiplies_qr(M, d)
    iter_RRt_cost = d * d * d
    iter_svd_cost = _nmultiplies_svd(d, d)
    iter_reweight_cost = N * d + M * d
    iter_update_x_y_cost = (N * d * d) + (M * d * d)
    iter_cost = (iter_qr_cost + iter_RRt_cost + iter_svd_cost +
                 iter_reweight_cost + iter_update_x_y_cost)
    return niters * iter_cost


def test_cooccur_sketches():
    print("test_cooccur_sketches")
    # so this doesn't have monotonically better acc as d increases; seems to
    # run into issues with d being a large fraction of D, possibly because
    # then it doesn't have many iterations and it's just zeroing out a ton of
    # the singular vectors

    N, M, D = 100, 80, 50
    # N, M, D = 100, 80, 160
    np.random.seed(1234)
    A = np.random.randint(5, size=(N, D)).astype(np.float32)
    B = np.random.randint(5, size=(D, M)).astype(np.float32)
    A -= np.mean(A)
    B -= np.mean(B)

    AB = A @ B
    orig_frob_sq = np.sum(AB * AB)

    # prev_normed_err = np.inf
    # for d in [4]:
    for d in (2, 4, 8, 16, 32):
        # A_hat, B_hat = fd_amm_sketches(A, B, d)
        A_hat, B_hat = cooccur_sketches(A, B, d)
        AB_hat = A_hat @ B_hat

        # print("fused mats shapes: ")
        # print(Ua.shape, SVTa.shape, Ub.shape, SVTb.shape)

        diffs = AB - AB_hat
        err_frob_sq = np.sum(diffs * diffs)
        normed_err_sq = err_frob_sq / orig_frob_sq
        print('d = {}, err = {:.5f}'.format(d, normed_err_sq))
        assert normed_err_sq < 1.
        # assert normed_err_sq < prev_normed_err
        # prev_normed_err = normed_err_sq


# ================================================================ main

# def main():
#     pass


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.3}".format(f)})
    # test_sketch_sq_sample()
    # test_svd_sketches()
    # test_fd_amm_sketches()
    # test_cooccur_sketches()
    test_rand_sketches()

    # # N = 1000
    # # N = 100
    # N = 20
    # # N = 10
    # M = 10
    # # M = 5

    # x = np.arange(N)
    # # x *= x
    # # x *= x
    # # x = np.sqrt(x)
    # # x = 1.1 ** x
    # # x = 1.15 ** x
    # x = 2 ** x
    # # print("x = ", x)
    # # idxs, weights = sample_varopt_1d(x, M)
    # idxs, weights = importance_sample(x, M)
    # y = x[idxs] * weights
    # xsum, ysum = x.sum(), y.sum()
    # # print("idxs = ", idxs)
    # print("vals = ", x[idxs])
    # print("weights = ", weights)
    # print("vals * weights", y)
    # # print("true sum, sample sum: ", xsum, ysum)
    # print("sum rel err: ", (xsum - ysum) / xsum)
