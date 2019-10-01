#!/bin/env/python

import abc
import numpy as np
from sklearn.utils.extmath import randomized_svd

KEY_NMULTIPLIES = 'muls'


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
        if D < self.d:
            raise InvalidParametersException(
                'D < d: {} < {}'.format(D, self.d))
        return self.call(np.copy(A), np.copy(B))  # guarantee A, B unchanged

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        assert not (fixedA and fixedB)  # this would be stupid, so fail fast
        sketch_nmuls = self._get_nmuls(A.shape[0], A.shape[1], B.shape[1],
                                       self.d, fixedA=fixedA, fixedB=fixedB)
        return {KEY_NMULTIPLIES: sketch_nmuls + _nmultiplies_matmul(A, B)}

    def _get_nmuls(self, N, D, M, d, fixedA=False, fixedB=False):
        pass


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


class SvdSketch(SketchedMatmul):
    __slots__ = 'd niters Ua SVTa Ub SVTb'.split()

    def __init__(self, d, niters=5):
        self.d = d
        self.niters = niters
        self.Ua = None
        self.SVTa = None
        self.Ub = None
        self.SVTb = None

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


# ================================================================ samplings

def _compute_dim_scores(A, B, A_col_norms=None, B_row_norms=None):
    if A_col_norms is None:
        A_col_norms = np.linalg.norm(A, axis=0)
    if B_row_norms is None:
        B_row_norms = np.linalg.norm(B, axis=1)
    return A_col_norms * B_row_norms


def sketch_sq_sample(A, B, d):
    scores = _compute_dim_scores(A, B)  # TODO uncomment after debug
    # scores = np.ones(A.shape[1]) # TODO rm after debug
    probs = scores / np.sum(scores)

    D = A.shape[1]
    keep_idxs = np.random.choice(D, size=d, p=probs)
    # keep_idxs = np.random.choice(D, size=d, p=probs, replace=False)  # TODO rm
    # keep_idxs = np.random.choice(D, size=d, replace=False)  # TODO rm
    # keep_idxs = np.arange(D-1)  # TODO rm
    # keep_idxs = np.arange(1, D)  # TODO rm
    # keep_idxs = np.arange(D)  # TODO rm

    weights = np.sqrt(d * probs)  # what the paper says; huge errors
    # weights = np.sqrt(D * probs)  # slightly less bad
    # weights = np.sqrt(np.sqrt(d * probs))
    # weights = np.ones(D)
    A = np.copy(A) / weights
    B = np.copy(B) / weights.reshape(-1, 1)

    # return np.copy(A[:, keep_idxs]), np.copy(B[keep_idxs])
    return A[:, keep_idxs], B[keep_idxs]
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


# ================================================================ FD-amm

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


def fd_amm_sketches(A, B, d):
    # print("A shape: ", A.shape)
    # print("B shape: ", B.shape)
    G = np.hstack((A.T, B))   # D x (N + M)
    H = frequent_directions(G, d)
    assert H.shape == (d, A.shape[0] + B.shape[1])
    C = H[:, :A.shape[0]]  # d x N
    D = H[:, A.shape[0]:]  # d x M
    return C.T, D

# TODO this is runtime for fast-fd, not FD
def _nmultiplies_frequent_directions(N, D, d):
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
    # test_sketch_sq_sample()
    # test_svd_sketches()
    # test_fd_amm_sketches()
    test_cooccur_sketches()
