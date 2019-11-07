#!/bin/env/python

from __future__ import division

import numpy as np
import numba

from .utils import top_k_idxs


# ================================================================ eigenvecs

# @numba.jit(nopython=True)  # don't jit since take like 2.5s
# def top_principal_component(X, niters=50, return_eigenval=False,
def top_principal_component(X, niters=100, return_eigenval=False,
                            momentum=.9, nguesses=32, learning_rate=1.,
                            # allow_materialize=False):
                            allow_materialize_XtX=True):
    N, D = X.shape
    X = X.astype(np.float32)
    X = X - X.mean(axis=0)

    if nguesses > 1:
        V = np.random.randn(D, nguesses).astype(X.dtype)
        V /= np.linalg.norm(V, axis=0)
        # norms = np.sqrt((V * V).sum(axis=0))
        # V /= norms
        prods = X.T @ (X @ V)
        new_norms = np.linalg.norm(prods, axis=0)
        # new_norms_sq = (prods * prods).sum(axis=0)
        v = V[:, np.argmax(new_norms)]
        # v = V[:, np.argmax(new_norms_sq)]
        # print("picking v = ", v)
    else:
        v = np.random.randn(D).astype(X.dtype)
    # v = np.ones(D, dtype=np.float32)

    v = v.astype(np.float32)
    prev_v = np.zeros_like(v)
    v_momentum = np.zeros_like(v)
    v /= (np.linalg.norm(v) + 1e-20)

    materialize_cost = N * D * D
    iter_cost_no_materialize = 2 * N * D
    iter_cost_materialize = D * D

    materialize = (materialize_cost + (niters * iter_cost_materialize) <
                   (niters * iter_cost_no_materialize))
    materialize = materialize and allow_materialize_XtX
    if materialize:
        scaleby = np.max(np.linalg.norm(X, axis=0))
        X *= 1. / scaleby  # precondition by setting largest variance to 1
        XtX = X.T @ X

    for i in range(niters):
        if materialize:
            v = XtX @ v
        else:
            v = X.T @ (X @ v)
        v *= 1. / (np.linalg.norm(v) + 1e-20)
        # v_momentum = .9 * v_momentum + .5 * (v - prev_v)
        # v_momentum = (.9 * v_momentum + (v - prev_v)).astype(np.float32)
        v_momentum = momentum * v_momentum + learning_rate * (v - prev_v)
        v += v_momentum
        prev_v = v
        # if i % 5 == 0:
        #     print("v: ", v)

    v /= (np.linalg.norm(v) + 1e-20)
    if return_eigenval:
        new_v = X.T @ (X @ v)
        lamda = np.linalg.norm(new_v)
        return v, lamda
    return v


def top_principal_component_v1(X, init='gauss', niters=100,
                               return_eigenval=False, batch_sz=-1,
                               momentum=.9, nguesses=32, verbose=0):
    N, D = X.shape
    X = X.astype(np.float32)
    # Z = X - X.mean(axis=0)

    if nguesses is not None and nguesses > 1:
        assert init == 'gauss'

    if init == 'ones':
        v = np.ones(D, dtype=X.dtype)
    elif init == 'gauss':
        if nguesses > 1:
            V = np.random.randn(D, nguesses).astype(X.dtype)
            V /= np.linalg.norm(V, axis=0)
            prods = X.T @ (X @ V)
            new_norms = np.linalg.norm(prods, axis=0)
            # print("new_norms: ", new_norms)
            # assert np.min(eigenvals > -.001)  # should be nonneg
            v = V[:, np.argmax(new_norms)]
            # print("picking v = ", v)
        else:
            v = np.random.randn(D).astype(X.dtype)
    elif init == 'variance':
        v = X.var(axis=0)
    else:
        v = init  # can also pass in raw vector to initialize it with

    if batch_sz < 1:
        # batch_sz = min(2048, N)
        # batch_sz = min(N, max(2048, N // 4))
        # batch_sz = N // 4
        batch_sz = N
    nbatches = int(np.ceil(N / batch_sz))

    prev_v = np.zeros_like(v)
    v_momentum = np.zeros_like(v)
    v /= (np.linalg.norm(v) + 1e-20)

    for i in range(niters):
        v = X @ v
        # v /= (np.linalg.norm(v) + 1e-20)
        v = X.T @ v
        v /= (np.linalg.norm(v) + 1e-20)
        # v_momentum = .9 * v_momentum + .5 * (v - prev_v)
        v_momentum = .9 * v_momentum + (v - prev_v)
        # v_momentum = .95 * v_momentum + (v - prev_v)
        v += v_momentum
        prev_v = v
        if (verbose > 0) and (i % 5 == 0):
            print("----")
            print("mom: ", v_momentum)
            print("v:   ", v)
            # print("v, prev_v dot prod: ", v.T @ prev_v)

    # for i in range(niters):
    #     perm = np.random.permutation(nbatches)
    #     for b in range(nbatches):
    #         use_b = perm[b]  # shuffle order of batches across iters
    #         start_idx = use_b * batch_sz
    #         end_idx = min(N, start_idx + batch_sz)

    #         Zb = Z[start_idx:end_idx]
    #         Xb = X[start_idx:end_idx]
    #         # print("v: ", v)
    #         # print("Z shape", Z.shape)
    #         # print("X shape", X.shape)
    #         # print("X @ v shape", (X @ v).shape)
    #         # update based on Adaptive Synaptogenesis Constructs Neural Codes
    #         # That Benefit Discrimination, theorem 1; could also use Oja's rule
    #         # v += ((Z - v) * (X @ v).reshape(-1, 1)).mean(axis=0)

    #         # v += ((Zb - v) * (Xb @ v).reshape(-1, 1)).sum(axis=0)

    #         dv = ((Zb - v) * (Xb @ v).reshape(-1, 1)).mean(axis=0)
    #         # dv /= np.linalg.norm(dv)

    #         v += dv

    #         # v_momentum = .5 * v_momentum + .5 * dv
    #         # v += v_momentum

    #         # v += dv + v_momentum
    #         v /= (np.linalg.norm(v) + 1e-20)
    #         # v += v_momentum + dv
    #         v += v_momentum
    #         v /= (np.linalg.norm(v) + 1e-20)

    #         v_momentum = .8 * v_momentum + .5 * (v - prev_v)
    #         # v_momentum = .9 * v_momentum + .1 * dv
    #         # v_momentum = .9 * v_momentum + (v - prev_v)
    #         # v_momentum = .5 * v_momentum + .5 * dv

    #         if i % 5 == 0:
    #             print("----")
    #             print("v_momentum: ", v_momentum)
    #             print("prev_v:     ", prev_v)
    #             print("v:          ", v)

    #         prev_v[:] = v
    #         # v_momentum = .1 * dv

    v /= (np.linalg.norm(v) + 1e-20)
    if return_eigenval:
        new_v = X.T @ (X @ v)
        lamda = np.linalg.norm(new_v)
        return v, lamda
    return v


def power_iteration(A, niters=5, init='ones', return_eigenval=False):
    if init == 'ones':
        v = A.sum(axis=0)
    elif init == 'gauss':
        v = np.random.randn(A.shape[1]).astype(A.dtype)
    else:
        v = init  # can also pass in raw vector to initialize it with

    for i in range(niters):
        v /= (np.linalg.norm(v) + 1e-20)
        v = (A * v).mean(axis=1)

    lamda = np.linalg.norm(v)
    v /= (lamda + 1e-20)
    if return_eigenval:
        return v, lamda
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
    # np.random.seed(1234)
    # np.random.seed(6)
    # N, D = 20, 6
    N, D = 10000, 128
    # N, D = 1000, 128
    # N, D = 1000, 512
    # N, D = 10000, 64
    # N, D = 10000, 8
    # N, D = 10000, 10
    X = np.random.randn(N, D).astype(np.float32)
    # greedy_eigenvector_threshold(X, 3)
    # greedy_eigenvector_threshold(X, 3, sample_how='deterministic')
    # greedy_eigenvector_threshold(X, 3, sample_how='importance')
    # greedy_eigenvector_threshold(X, 3, use_corr=True)

    from sklearn.decomposition import PCA
    # # Z = X - X.mean(axis=0)
    # # pca = PCA(n_components=D).fit(X.T @ X)
    # pca = PCA(n_components=D).fit(X)
    # eigenvecs = pca.components_
    # print("PCA components:", eigenvecs)
    # print("PCA singular vals:", pca.singular_values_)
    # v, lamda = top_principal_component(X, return_eigenval=True, init='gauss')
    # print("v: ", v)
    # print("v * eigenvecs: ", (eigenvecs * v).sum(axis=1))

    from sklearn.decomposition import PCA
    import time

    # pca = PCA(n_components=D)
    # pca = PCA(n_components=D, svd_solver='full')  # TODO rm
    pca = PCA(n_components=1, svd_solver='full')  # TODO rm
    # pca = PCA(n_components=1, svd_solver='randomized')
    t = time.perf_counter()
    pca.fit(X)
    nsecs = time.perf_counter() - t
    print("pca time (s): ", nsecs)

    t = time.perf_counter()
    v = top_principal_component(X)
    nsecs = time.perf_counter() - t
    print("our time (s): ", nsecs)

    print("v * eigenvecs: ", (pca.components_ * v).sum(axis=1)[:5])
    # print("cossim between vecs: ", pca.components_ @ v)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.3}".format(f)},
                        linewidth=100)
    main()
