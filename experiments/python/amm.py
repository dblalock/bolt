#!/bin/env/python

import numpy as np


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


def main():
    N, M, D = 100, 50, 200
    np.random.seed(1234)
    A = np.random.randint(5, size=(N, D)).astype(np.float32)
    B = np.random.randint(5, size=(D, M)).astype(np.float32)

    AB = A @ B
    orig_frob_sq = np.sum(AB * AB)

    # normed_errs = []
    prev_normed_err = np.inf
    for d in (10, 20, 30, 40, 50):
        A_hat, B_hat = sketch_sq_sample(A, B, d)
        # A_hat, B_hat = sketch_sq_deterministic(A, B, d)
        AB_hat = A_hat @ B_hat
        diffs = AB - AB_hat
        err_frob_sq = np.sum(diffs * diffs)
        normed_err_sq = err_frob_sq / orig_frob_sq
        print('d = {}, err = {:.3f}'.format(d, normed_err_sq))
        assert normed_err_sq < prev_normed_err
        prev_normed_err = normed_err_sq
        # normed_errs.append(normed_err_sq)


if __name__ == '__main__':
    main()
