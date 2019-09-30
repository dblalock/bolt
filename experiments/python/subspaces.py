#!/bin/env/python

from __future__ import division

import numpy as np

from .utils import top_k_idxs


# ================================================================ eigenvecs

def power_iteration(A, niters=5, init='ones', return_eigenval=False):
    if init == 'ones':
        v = A.sum(axis=0)
    elif init == 'gauss':
        v = np.random.randn(A.shape[1]).astype(A.dtype)
    else:
        v = init  # can also pass in raw vector to initialize it with

    for i in range(niters):
        v /= (np.linalg.norm(v) + 1e-20)
        v = (A * v).sum(axis=1)

    if return_eigenval:
        return v, np.linalg.norm(v)
    #     return v, np.linalg.norm(v) / np.linalg.norm(old_v)
    return v


def greedy_eigenvector_threshold(X, subspace_len, sample_how='deterministic',
                                 stats_mat='cov', npower_iters=5,
                                 nsubspaces=-1):
    # nsubspaces=-1, col_stds=None):
    assert sample_how in ('deterministic', 'importance')
    # print("nsubspaces: ", nsubspaces)
    # print("X.shape", X.shape)

    # rm all-zero columns; if whole thing is zero, just return original order
    # keep_cols = (X != 0).sum(axis=0) != 0
    # nnz_cols = np.sum(keep_cols) != 0
    # orig_all_idxs = np.arange(X.shape[1])
    # if nnz_cols == 0:
    #     return orig_all_idxs
    # else:
    #     orig_D = X.shape[1]
    #     X = X[:, keep_cols]

    # numpy funcs for corr, cov are too quick to create nans, so compute stats
    # manually
    N, D = X.shape
    if stats_mat == 'cov':
        X = (X - X.mean(axis=0)) / np.sqrt(N)
        cov = X.T @ X
    elif stats_mat == 'corr':
        X = (X - X.mean(axis=0)) / (np.linalg.norm(X, axis=0) + 1e-14)
        cov = X.T @ X
    else:
        assert X.shape[0] == X.shape[1]  # using X as the cov/corr mat
        cov = X

    # if col_stds is None:
    #     col_stds = np.std(cov, axis=0) + 1e-14

    if nsubspaces is None or nsubspaces < 0:
        nsubspaces = int(np.ceil(D / subspace_len))

    all_idxs = np.arange(D)
    if nsubspaces == 1:
        return all_idxs

    # find the indices to add to the next subspace
    v = power_iteration(cov, niters=npower_iters)
    if sample_how == 'deterministic':
        idxs = top_k_idxs(np.abs(v), subspace_len, smaller_better=False)
    elif sample_how == 'importance':
        probs = np.abs(v) + 1e-14
        probs /= np.sum(probs)
        idxs = np.random.choice(all_idxs, size=subspace_len,
                                p=probs, replace=False)

    # remove the indices we selected, and recurse
    mask = np.ones(D, dtype=np.bool)
    mask[idxs] = False
    cov = cov[mask][:, mask]
    # col_stds = col_stds[mask]
    rest_of_perm = greedy_eigenvector_threshold(
        cov, subspace_len, sample_how=sample_how, stats_mat=None,
        npower_iters=npower_iters, nsubspaces=nsubspaces - 1)
    #     nsubspaces=nsubspaces - 1, col_stds=col_stds)

    # convert indices from recursive call (which are in a different subspace
    # since we excluded some indices) back to the original space
    rest_of_perm = all_idxs[mask][rest_of_perm]  # child call using subspace

    # perm = np.array(list(idxs) + list(rest_of_perm))
    perm = np.r_[idxs, rest_of_perm]
    # if orig_D > D:  # we removed some zero cols at the beginning
    #     perm = orig_all_idxs[keep_cols][perm]

    if len(set(perm)) != len(perm):  # TODO rm after debug
        print("nsubspaces, subspace_len: ", nsubspaces, subspace_len)
        print("size of set(all_idxs)", len(set(all_idxs)))
        print("size of set(perm)", len(set(perm)))
        assert len(set(perm)) == len(perm)
        # import sys; sys.exit()

    return perm

    # v = 'ones'  # in subseqent iters, will init with prev eigenva

    # zero_cols =

    # TODO ideally actually pull rows/cols out of cov to create a smaller
    # matrix, so that later subspaces have less work to do; issue here is
    # that it makes keeping track of what the indices mean pretty ugly


    # if nsubspaces == 1:
    #     return all_idxs

    # mask = np.zeros(D, dtype=np.bool)
    # perm = []
    # for m in range(nsubspaces - 1):
    #     v = power_iteration(cov, niters=npower_iters)
    #     if sample_how == 'deterministic':
    #         idxs = top_k_idxs(np.abs(v), subspace_len, smaller_better=False)
    #     elif sample_how == 'importance':
    #         probs = np.abs(v)
    #         probs /= np.sum(probs) + 1e-14

    #         # # TODO rm after debug
    #         # nnz = np.sum(probs > 0)
    #         # # if nnz < subspace_len:
    #         # print("m: {}/{}".format(m + 1, nsubspaces))
    #         # print("D:", D)
    #         # print("subspace_len:", subspace_len)
    #         # print("nnz:", nnz)

    #         try:
    #             idxs = np.random.choice(all_idxs, size=subspace_len,
    #                                     p=probs, replace=False)
    #         except ValueError:
    #             missing_idxs = set(all_idxs) - set(perm)
    #             perm += list(missing_idxs)
    #             break

    #     perm += list(idxs)
    #     # print("adding {} idxs".format(len(idxs)))
    #     # print("new len(perm)", len(perm))

    #     # rm selected indices from future consideration
    #     mask[:] = True
    #     mask[idxs] = False
    #     cov = cov[mask, mask]

    #     # # rm the selected indices from future consideration
    #     # mask[:] = False
    #     # mask[idxs] = True
    #     # # print("cov.shape: ", cov.shape)
    #     # # # print("mask.shape: ", mask.shape)
    #     # # print("idxs: ", idxs)
    #     # # print("mask: ", mask)
    #     # # print("cov[mask]\n", cov[mask])
    #     # # print("cov[:, mask]\n", cov[:, mask])
    #     # # cov[mask, mask] = 0
    #     # # print("nnz in mask: ", np.sum(mask != 0))
    #     # cov[mask] = 0
    #     # cov[:, mask] = 0
    #     # # print("cov[mask]\n", cov[mask])
    #     # # print("cov[:, mask]\n", cov[:, mask])
    #     # # print("idxs: ", idxs)
    #     # # print("cov[idxs].sum(axis=1)", cov[idxs].sum(axis=1))
    #     # # print("cov[:, idxs].sum(axis=0)", cov[:, idxs].sum(axis=0))
    #     # # print("nnz cols in cov: ", np.sum(cov.sum(axis=0) != 0))
    #     # # assert np.all(cov[mask] == 0)
    #     # # assert np.all(cov[:, mask] == 0)

    # # add whatever indices are left over to last subspace; doing it this way
    # # both saves us work and avoids breaking things when some columns are 0
    # # as a result of earlier padding to multiple of subspace_len
    # missing_idxs = set(all_idxs) - set(perm)

    # if len(set(perm)) != len(perm):  # TODO rm after debug
    #     print("nsubspaces, subspace_len: ", nsubspaces, subspace_len)
    #     print("size of set(all_idxs)", len(set(all_idxs)))
    #     print("size of set(perm)", len(set(perm)))
    #     print("number of missing_idxs", len(missing_idxs))
    #     # assert len(set(perm)) == len(perm)
    #     import sys; sys.exit()

    # perm += list(missing_idxs)

    # return np.array(perm)

    # return all_idxs[::-1] # TODO rm after debug
    # return np.roll(all_idxs, 1) # TODO rm after debug
    # return all_idxs # TODO rm after debug


# ================================================================ main

def main():
    N, D = 20, 6
    X = np.random.randn(N, D).astype(np.float32)
    greedy_eigenvector_threshold(X, 3)
    greedy_eigenvector_threshold(X, 3, sample_how='deterministic')
    greedy_eigenvector_threshold(X, 3, sample_how='importance')
    greedy_eigenvector_threshold(X, 3, use_corr=True)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.3}".format(f)})
    main()
