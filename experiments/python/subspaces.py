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


# def ksparse_pca(X, ncomponents, k, algo='anydims'):
# def ksparse_pca(X, ncomponents, k, algo='noreuse'):
def ksparse_pca_v1(X, ncomponents, k, algo='1uniq'):
    N, D = X.shape
    k = int(k)
    assert k < D  # TODO run dense randomized PCA to handle this case
    if algo == 'noreuse':
        assert ncomponents * k <= D  # TODO allow dims to be included >1 time

    from sklearn.linear_model import OrthogonalMatchingPursuit
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
    if algo == '1uniq':
        assert k > 1
        omp_initial = OrthogonalMatchingPursuit(
            n_nonzero_coefs=k - 1, fit_intercept=False)
        omp_final = OrthogonalMatchingPursuit(
            n_nonzero_coefs=1, fit_intercept=False)

    X = np.asfarray(X)  # we'll be taking subsets of columns a lot
    X_res = np.copy(X)

    # allowed_idxs = set(np.arange(D))
    allowed_idxs = np.arange(D)
    # all_used_idxs = set()

    V = None
    for i in range(ncomponents):
        # compute ideal projection, and resulting latent values
        v = top_principal_component(X_res).reshape(D, 1)
        if i > 0:
            # gram-schmidt to orthogonalize; we don't get to use this exact
            # vector anyway, so we don't care too much about numerical issues;
            # also, principal component of residuals should be in a subspace
            # that's orthogonal to V already, so might be able to prove this
            # step isn't even necessary
            prods = (V.T @ v).ravel()   # (D x i+1).T @ (D x 1) = i+1 x 1
            # print("prods shape: ", prods.shape)
            # print("V shape: ", V.shape)
            # print("v shape: ", v.shape)
            # print("projections shape: ", (V * prods).shape)
            v -= (V * prods).sum(axis=1, keepdims=True)
            # V = np.hstack((V, v))
            # V, R = np.linalg.qr(V)
            # v = V[-1]
        h = X_res @ v  # N x 1

        # compute sparse version of this ideal projection
        # if False:
        if algo == 'anydims':
            v = omp.fit(X, h).coef_
        elif algo == '1uniq':  # 1 new idx -> possible to make orthogonal
            assert k > 1
            if i == 0:
                v = omp.fit(X, h).coef_
                # used_idxs = np.where(v != 0)[0]
                # all_used_idxs += set(used_idxs)
            else:
                # compute k-1 sparse v
                v = omp_initial.fit(X, h).coef_.ravel()
                initial_nonzero_idxs = np.where(v != 0)[0]

                # now find last zero to add, from set that have never been used
                h_res = h - (X @ v)

                use_allowed_idxs = set(allowed_idxs) - set(initial_nonzero_idxs)
                use_allowed_idxs = np.array(sorted(list(use_allowed_idxs)))
                X_subs = X[:, use_allowed_idxs]

                soln = omp_final.fit(X_subs, h_res).coef_.ravel()
                new_nonzero_idx = use_allowed_idxs[np.where(soln != 0)[0][0]]

                # now take union of all these idxs to get nonzero idxs to use
                use_idxs = list(initial_nonzero_idxs) + [new_nonzero_idx]
                use_idxs = np.array(use_idxs)
                # print("use_idxs", use_idxs)

                # given nonzero idxs, least squares to get v
                X_subs = X[:, use_idxs]
                soln, _, _, _ = np.linalg.lstsq(X_subs, h, rcond=None)

                v = np.zeros(D)
                v[use_idxs] = soln.ravel()
        else:  # dims outright can't be reused
            assert algo == 'noreuse'
            X_subs = X[:, allowed_idxs]
            assert len(allowed_idxs) >= k
            soln = omp.fit(X_subs, h).coef_
            v = np.zeros(D)
            v[allowed_idxs] = soln

        v = v.reshape(-1, 1)
        v /= np.linalg.norm(v)
        assert np.sum(v != 0) == k

        # update V, ensuring that it remains orthonormal
        # TODO the issue with this is that there doesn't necessarily *exist*
        # a k-sparse vector that's orthogonal to all others picked so far; we
        # could solve this by requiring dk <= D and making it impossible to
        # select the same input dimension twice; that's more restrictive than
        # is strictly needed though; what would be really nice is just writing
        # our own OMP that you can tell to not select certain idxs, because
        # that would create a combination that can't be made orthogonal;
        # probably adapt https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/linear_model/omp.py#L407
        if i > 0:
            # if dims_can_be_reused:
            if algo != 'noreuse':

                nnz_idxs = np.where(v != 0)[0]
                assert len(nnz_idxs) <= k
                V_subs = V[nnz_idxs]
                v_subs = v[nnz_idxs]

                # niters_ortho = 1000
                niters_ortho = 100
                for it in range(niters_ortho):

                    if False:
                        prods = (V.T @ v).ravel()
                        # print("prods shape: ", prods.shape)
                        # print("V shape: ", V.shape)
                        # print("v shape: ", v.shape)
                        # print("projections shape: ", (V * prods).shape)
                        v -= (V * prods).sum(axis=1, keepdims=True)
                        v = v.ravel()
                        zero_out_idxs = np.argsort(np.abs(v))[:-k]
                        # keep_idxs = np.argsort(np.abs(v))[-k:]
                        # print("i, it: ", i, it)
                        # print(f"zeroing out {len(zero_out_idxs)} / {D} indices")
                        # print("nnz before zeroing: ", np.sum(v != 0))
                        # old_v = v
                        # v = np.zeros(D)
                        # v[keep_idxs] = old_v[keep_idxs]

                        v[zero_out_idxs] = 0
                        nnz = np.sum(v != 0)
                        # print("nnz: ", nnz)
                        # print("len v:", len(v))
                        # print("v", v)
                        assert nnz <= k
                        v /= np.linalg.norm(v)
                        v = v.reshape(-1, 1)
                    else:
                        prods = (V_subs.T @ v_subs).ravel()
                        v_subs -= (V_subs * prods).sum(axis=1, keepdims=True)
                        # v_subs = v_subs.ravel()

                    if np.max(np.abs(prods)) < 1e-5:  # TODO add tol param
                        # print("breaking at iter: ", it)
                        break  # pretty converged

            v = v.ravel()
            v[:] = 0
            v[nnz_idxs] = v_subs.ravel()
            v /= np.linalg.norm(v)
            v = v.reshape(-1, 1)

        if algo in ('noreuse', '1uniq'):
            used_idxs = np.where(v != 0)[0]
            # used_idxs = [np.argmax(np.abs(v))]  # only eliminate 1 idx
            allowed_idxs = set(allowed_idxs) - set(used_idxs)
            allowed_idxs = np.array(sorted(list(allowed_idxs)))

        if i > 0:
            V = np.hstack((V, v))
        else:
            V = v

        # now update X_res; residuals from best linear approx of input given H
        H = X_res @ V
        W, _, _, _ = np.linalg.lstsq(H, X, rcond=None)
        X_res = X - (H @ W)

    return V


# these are just for debugging
def _to_sparse(x):
    x = x.ravel()
    idxs = np.where(x != 0)[0]
    vals = x[idxs]
    idxs = idxs.reshape(-1, 1)
    vals = vals.reshape(-1, 1)
    # print("idxs: ", idxs)
    # print("vals: ", vals)
    return np.hstack((idxs, vals))


def _to_sparse_cols(A):
    ars = [_to_sparse(A[:, j])[np.newaxis, ...]
           for j in range(A.shape[1])]
    return "\n".join([str(ar) for ar in ars])
    # return np.concatenate(vecs, axis=0)


def ksparse_pca(X, ncomponents, k):
    N, D = X.shape
    k = int(k)
    assert k < D  # TODO run dense randomized PCA to handle this cases
    X = np.asfarray(X)  # we'll be taking subsets of columns a lot
    X_res = np.copy(X)

    from sklearn.linear_model import OrthogonalMatchingPursuit
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)

    idx_counts = np.zeros(D, dtype=np.int)

    V = None
    for i in range(ncomponents):
        v = top_principal_component(X_res).reshape(D, 1)
        if i > 0:
            # gram-schmidt to orthogonalize; we don't get to use this exact
            # vector anyway, so we don't care too much about numerical issues;
            # also, principal component of residuals should be in a subspace
            # that's orthogonal to V already, so might be able to prove this
            # step isn't even necessary
            prods = (V.T @ v).ravel()   # (D x i+1).T @ (D x 1) = i+1 x 1
            v -= (V * prods).sum(axis=1, keepdims=True)
        h = X_res @ v

        # compute sparse version of this ideal projection
        allowed_idxs = idx_counts < k
        X_subs = X[:, allowed_idxs]
        assert allowed_idxs.sum() >= k
        soln = omp.fit(X_subs, h).coef_
        v = np.zeros(D)
        v[allowed_idxs] = soln

        nnz_idxs = v != 0
        v = v.reshape(-1, 1)
        v /= np.linalg.norm(v)
        assert np.sum(v != 0) == k



        # TODO this is broken because having no dim used more than k times
        # isn't actually a sufficient condition to ensure that cols of V
        # can be made orthogonal; need to write our own OMP that can take
        # in existing nnz pattern of V and not include dims that would result
        # in too many linearly indep cols in that subspace



        # update idx_counts
        idx_counts[nnz_idxs] += 1

        # make v orthogonal to existing cols in V
        if V is None:
            V = v
            continue
        V_subs = V[nnz_idxs].copy()
        nonzero_cols = V_subs.sum(axis=0) != 0
        if np.sum(nonzero_cols) < 1:  # already orthogonal to existing V
            V = np.hstack((V, v))
            continue
        V_subs_orig = V_subs.copy()
        V_subs = V_subs[:, nonzero_cols]
        # V_subs, _ = np.linalg.qr(V_subs)


        debug = i == 7


        v_subs = v[nnz_idxs].copy()
        niters_ortho = 100 if not debug else 1
        v_orig = v.copy()
        v_subs_orig = v_subs.copy()
        for it in range(niters_ortho):
            prods = (V_subs.T @ v_subs).ravel()
            projections = (V_subs * prods).sum(axis=1, keepdims=True)
            # v_subs -= .999 * projections
            v_subs -= projections
            V_subs = V_subs[:, prods != 0]

            # if debug:
            #     print("V_subs:\n", V_subs)
            #     print("projections: ", projections)
            #     # print("v_subs: ", projections)

            # SELF: issue here is that cols of V_subs are not necessarily
            # orthogonal, so projections can actually overcorrect and have
            # exactly the wrong component come to dominate

            v_subs /= np.linalg.norm(v_subs)
            if np.max(np.abs(prods)) < 1e-5:  # TODO add tol param
                # print("breaking at iter: ", it)
                break  # pretty converged
        if it == niters_ortho - 1:
            print(f"k={k}, it={it}")
            print(f"FAILED to get component {i} orthogonal")
            print("prods:\n", prods)
            # print("v before gram-schmidt:")
            # print(_to_sparse(v_orig))
            # print("V with nonzeros in subspace: ")
            # V_subset = V[:, prods != 0]
            # print("V_subset shape:", V_subset.shape)
            # print(_to_sparse_cols(V_subset))
            # # print(V[:, prods != 0])
            # print("v:")
            # print(_to_sparse(v))

            print("projections:", projections)
            print("V_subs_orig\n", V_subs_orig)
            print("v_subs_orig\n", v_subs_orig)
            print("V_subs:\n", V_subs[:, prods != 0])
            print("v_subs:", v_subs)
            import sys; sys.exit()

            # print("got to ortho iteration: ", it)
            # nonzero_count_idxs = np.where(idx_counts)[0]
            # print("idx counts:\n", np.array(list(zip(nonzero_count_idxs, idx_counts[nonzero_count_idxs]))).T)
            # print("picked idxs: ", np.where(nnz_idxs)[0])
        v = v.ravel()
        v[:] = 0
        v[nnz_idxs] = v_subs.ravel()
        v /= np.linalg.norm(v)
        v = v.reshape(-1, 1)
        V = np.hstack((V, v))

        # now update X_res; residuals from best linear approx of input given H
        H = X_res @ V
        W, _, _, _ = np.linalg.lstsq(H, X, rcond=None)
        X_res = X - (H @ W)

    return V


def debug_orthogonalize():
    # V = np.array([[0.0, 0.0, 0.0],
    #               [-0.72, -0.367, 0.55],
    #               [-0.463, 0.482, 0.0],
    #               [-0.391, -0.457, -0.797]])
    # v = np.array([[-0.243],
    #               [-0.705],
    #               [-0.427],
    #               [-0.511]])
    V = np.array([[0.759, 0.506, 0.41],
                  [-0.58, 0.811, 0.0733],
                  [0.0, 0.0, 0.0],
                  [-0.296, -0.294, 0.909]])
    v = np.array([[0.729],
                 [-0.547],
                 [0.261],
                 [-0.318]])
    print("V:\n", V)
    print("v:\n", v)
    V /= np.linalg.norm(V, axis=0)
    print("V norms: ", np.linalg.norm(V, axis=0))

    for it in range(1):
        prods = (V.T @ v).ravel()
        print("prods: ", prods)
        projections = (V * prods).sum(axis=1, keepdims=True)
        print("projections:\n", projections)
        v -= projections
        v /= np.linalg.norm(v)

        # print("V:\n", V)
        print("new v:\n", v)

        # print("new prods: ", prods)
        # prods = (V.T @ v).ravel()


# ================================================================ main

def main():
    # debug_orthogonalize(); return # TODO rm


    np.random.seed(12)
    # np.random.seed(6)
    # N, D = 20, 10
    # N, D = 10000, 128
    # N, D = 1000, 128
    # N, D = 1000, 512
    N, D = 10000, 64
    # N, D = 10000, 32
    # N, D = 10000, 16
    # N, D = 10000, 8
    # N, D = 10000, 10
    d = int(D / 4)

    # create X with low-rank structure
    # np.random.seed(123)
    X0 = np.random.randn(N, d).astype(np.float32)
    X1 = np.random.randn(d, D).astype(np.float32)
    X = X0 @ X1
    X += np.random.randn(N, D).astype(np.float32) * .1

    # X = np.random.randn(N, D).astype(np.float32)
    # greedy_eigenvector_threshold(X, 3)
    # greedy_eigenvector_threshold(X, 3, sample_how='deterministic')
    # greedy_eigenvector_threshold(X, 3, sample_how='importance')
    # greedy_eigenvector_threshold(X, 3, use_corr=True)

    # k = 1  # k = 1 is really interesting; corresponds to just subsampling cols
    # k = 2
    # k = 4
    # k = 6
    k = 8
    k = min(k, int(D / d))
    V = ksparse_pca(X, d, k)
    H = X @ V
    W, _, _, _ = np.linalg.lstsq(H, X, rcond=None)
    X_res = X - (H @ W)
    print("X sq frob norm: ", np.sum(X * X))
    print("X res sq frob norm: ", np.sum(X_res * X_res))
    # print("nnz in V cols: ", (V != 0).sum(axis=0))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=d).fit(X)
    # print("pca explained variance: ", pca.explained_variance_)
    V2 = pca.components_.T
    H = X @ V2
    W, _, _, _ = np.linalg.lstsq(H, X, rcond=None)
    X_res = X - (H @ W)
    print("pca X res sq frob norm: ", np.sum(X_res * X_res))

    VtV = V.T @ V
    VtV2 = V2.T @ V2
    our_abs_offdiags = np.abs(VtV) - np.diag(np.diag(VtV))
    pca_abs_offdiags = np.abs(VtV2) - np.diag(np.diag(VtV2))
    print("our max abs off-diagonal, pca max abs off-diagonal:")
    print(np.max(our_abs_offdiags))
    print(np.max(pca_abs_offdiags))
    print("our mean abs off-diagonal, pca mean abs off-diagonal:")
    print(np.mean(our_abs_offdiags))
    print(np.mean(pca_abs_offdiags))

    # import matplotlib.pyplot as plt
    # import seaborn as sb
    # _, axes = plt.subplots(2)
    # # sb.heatmap(V.T @ V, ax=axes[0], cmap='RdBu')
    # # sb.heatmap(V2.T @ V2, ax=axes[1], cmap='RdBu')
    # sb.heatmap(V.T @ V, ax=axes[0])
    # sb.heatmap(V2.T @ V2, ax=axes[1])
    # # axes[0].imshow(V.T @ V, interpolation='nearest', cmap='RdBu')
    # # plt.colorbar(ax=axes[0])
    # # axes[0].imshow(V2.T @ V2, interpolation='nearest', cmap='RdBu')
    # # plt.colorbar(ax=axes[1])
    # axes[0].set_title("our V.T @ V")
    # axes[1].set_title("pca V.T @ V")
    # plt.tight_layout()
    # plt.show()

    # print("our V.T @ V: ", V.T @ V)
    # print("pca V.T @ V: ", V2.T @ V2)

    # # # Z = X - X.mean(axis=0)
    # # # pca = PCA(n_components=D).fit(X.T @ X)
    # # pca = PCA(n_components=D).fit(X)
    # # eigenvecs = pca.components_
    # # print("PCA components:", eigenvecs)
    # # print("PCA singular vals:", pca.singular_values_)
    # # v, lamda = top_principal_component(X, return_eigenval=True, init='gauss')
    # # print("v: ", v)
    # # print("v * eigenvecs: ", (eigenvecs * v).sum(axis=1))

    # from sklearn.decomposition import PCA
    # import time

    # # pca = PCA(n_components=D)
    # # pca = PCA(n_components=D, svd_solver='full')  # TODO rm
    # pca = PCA(n_components=1, svd_solver='full')  # TODO rm
    # # pca = PCA(n_components=1, svd_solver='randomized')
    # t = time.perf_counter()
    # pca.fit(X)
    # nsecs = time.perf_counter() - t
    # print("pca time (s): ", nsecs)

    # t = time.perf_counter()
    # v = top_principal_component(X)
    # nsecs = time.perf_counter() - t
    # print("our time (s): ", nsecs)

    # print("v * eigenvecs: ", (pca.components_ * v).sum(axis=1)[:5])
    # # print("cossim between vecs: ", pca.components_ @ v)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.3}".format(f)},
                        linewidth=100)
    main()
