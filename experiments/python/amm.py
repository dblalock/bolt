#!/bin/env/python

import numpy as np
from sklearn.utils.extmath import randomized_svd


# ================================================================ samplings

def _compute_dim_scores(A, B, A_col_norms=None, B_row_norms=None):
    if A_col_norms is None:
        A_col_norms = np.linalg.norm(A, axis=0)
    if B_row_norms is None:
        B_row_norms = np.linalg.norm(B, axis=1)
    return A_col_norms * B_row_norms


def sketch_sq_sample(A, B, d):
    scores = _compute_dim_scores(A, B)
    probs = scores / np.sum(scores)
    D = A.shape[1]
    keep_idxs = np.random.choice(D, size=d, p=probs)

    weights = np.sqrt(d * probs[keep_idxs])
    return A[:, keep_idxs] / weights, B[keep_idxs] / weights.reshape(-1, 1)


def sketch_sq_deterministic(A, B, d):
    scores = _compute_dim_scores(A, B)
    D = A.shape[1]
    keep_idxs = np.argsort(scores)[::-d]

    weights = np.sqrt(d * (1. / D))  # uniform prob
    return A[:, keep_idxs] / weights, B[keep_idxs] / weights


def test_sketch_sq_sample():
    N, M, D = 100, 50, 200
    np.random.seed(1234)
    A = np.random.randint(5, size=(N, D)).astype(np.float32)
    B = np.random.randint(5, size=(D, M)).astype(np.float32)

    AB = A @ B
    orig_frob_sq = np.sum(AB * AB)

    prev_normed_err = np.inf
    for d in (10, 20, 30, 40, 50):
        A_hat, B_hat = sketch_sq_sample(A, B, d)
        # A_hat, B_hat = sketch_sq_deterministic(A, B, d)
        AB_hat = A_hat @ B_hat
        diffs = AB - AB_hat
        err_frob_sq = np.sum(diffs * diffs)
        normed_err_sq = err_frob_sq / orig_frob_sq
        print('d = {}, err = {:.3f}'.format(d, normed_err_sq))
        assert normed_err_sq < 1.
        assert normed_err_sq < prev_normed_err
        prev_normed_err = normed_err_sq


# ================================================================ FD methods

def frequent_directions(A):
    pass


def fd_amm_sketches(A, B, d):
    pass


# ================================================================ TruncatedSVD

def svd_sketches(A, B, d, **kwargs):
    Ua, Sa, VTa = randomized_svd(A, n_components=d, **kwargs)
    Ub, Sb, VTb = randomized_svd(B, n_components=d, **kwargs)

    # print("truncated svd mat shapes:")
    # print(Ua.shape, Sa.shape, VTa.shape)
    # print(Ub.shape, Sb.shape, VTb.shape)

    return (Ua, np.diag(Sa) @ VTa), (Ub, np.diag(Sb) @ VTb)


def test_svd_sketches():
    N, M, D = 100, 80, 50
    np.random.seed(1234)
    A = np.random.randint(5, size=(N, D)).astype(np.float32)
    B = np.random.randint(5, size=(D, M)).astype(np.float32)

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


# ================================================================ main

# def main():
#     pass


if __name__ == '__main__':
    # test_sketch_sq_sample()
    test_svd_sketches()
