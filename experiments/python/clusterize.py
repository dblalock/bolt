#!/bin/env/python

import copy
import numpy as np
from functools import reduce

import numba
from sklearn.decomposition import PCA
from sklearn import linear_model

from . import subspaces as subs

from joblib import Memory
_memory = Memory('.', verbose=0)

# def bucket_id_to_new_bucket_ids(old_id):
#     i = 2 * old_id
#     return i, i + 1


class Bucket(object):
    __slots__ = 'N D id sumX sumX2 point_ids support_add_and_remove'.split()

    def __init__(self, D=None, N=0, sumX=None, sumX2=None, point_ids=None,
                 bucket_id=0, support_add_and_remove=False):
        # self.reset(D=D, sumX=sumX, sumX2=sumX2)
        # assert point_ids is not None
        if point_ids is None:
            assert N == 0
            point_ids = (set() if support_add_and_remove
                         else np.array([], dtype=np.int))

        self.N = len(point_ids)
        self.id = bucket_id

        # this is just so that we can store the point ids as array instead of
        # set, while still retaining option to run our old code that needs
        # them to be a set for efficient inserts and deletes
        self.support_add_and_remove = support_add_and_remove
        if support_add_and_remove:
            self.point_ids = set(point_ids)
        else:
            self.point_ids = np.asarray(point_ids)

        # figure out D
        if (D is None or D < 1) and (sumX is not None):
            D = len(sumX)
        elif (D is None or D < 1) and (sumX2 is not None):
            D = len(sumX2)
        assert D is not None
        self.D = D

        # figure out + sanity check stats arrays
        self.sumX = np.zeros(D, dtype=np.float32) if (sumX is None) else sumX
        self.sumX2 = np.zeros(D, dtype=np.float32) if (sumX2 is None) else sumX2 # noqa
        # print("D: ", D)
        # print("sumX type: ", type(sumX))
        assert len(self.sumX) == D
        assert len(self.sumX2) == D
        self.sumX = np.asarray(self.sumX).astype(np.float32)
        self.sumX2 = np.asarray(self.sumX2).astype(np.float32)

    def add_point(self, point, point_id=None):
        assert self.support_add_and_remove
        # TODO replace with more numerically stable updates if necessary
        self.N += 1
        self.sumX += point
        self.sumX2 += point * point
        if point_id is not None:
            self.point_ids.add(point_id)

    def remove_point(self, point, point_id=None):
        assert self.support_add_and_remove
        self.N -= 1
        self.sumX -= point
        self.sumX2 -= point * point
        if point_id is not None:
            self.point_ids.remove(point_id)

    def deepcopy(self, bucket_id=None):  # deep copy
        bucket_id = self.id if bucket_id is None else bucket_id
        return Bucket(
            sumX=np.copy(self.sumX), sumX2=np.copy(self.sumX2),
            point_ids=copy.deepcopy(self.point_ids), bucket_id=bucket_id)

    def split(self, X=None, dim=None, val=None, X_orig=None):
        id0 = 2 * self.id
        id1 = id0 + 1
        if X is None or self.N < 2:  # copy of this bucket + an empty bucket
            return (self.deepcopy(bucket_id=id0),
                    Bucket(D=self.D, bucket_id=id1))
        assert dim is not None
        assert val is not None
        assert self.point_ids is not None
        my_idxs = np.asarray(self.point_ids)

        # print("my_idxs shape, dtype", my_idxs.shape, my_idxs.dtype)
        X = X[my_idxs]
        X_orig = X if X_orig is None else X_orig[my_idxs]
        mask = X_orig[:, dim] < val
        not_mask = ~mask
        X0 = X[mask]
        X1 = X[not_mask]
        ids0 = my_idxs[mask]
        ids1 = my_idxs[not_mask]

        def create_bucket(points, ids, bucket_id):
            sumX = points.sum(axis=0) if len(ids) else None
            sumX2 = (points * points).sum(axis=0) if len(ids) else None
            # return Bucket(N=len(ids), D=self.D, point_ids=ids,
            return Bucket(D=self.D, point_ids=ids, sumX=sumX, sumX2=sumX2,
                          bucket_id=bucket_id)

        return create_bucket(X0, ids0, id0), create_bucket(X1, ids1, id1)

    def optimal_split_val(self, X, dim, possible_vals=None, X_orig=None,
                          return_possible_vals_losses=False):
        if self.N < 2 or self.point_ids is None:
            if return_possible_vals_losses:
                return 0, 0, np.zeros(len(possible_vals), dtype=X.dtype)
            return 0, 0
        # my_idxs = np.array(list(self.point_ids))
        my_idxs = np.asarray(self.point_ids)
        if X_orig is not None:
            X_orig = X_orig[my_idxs]
        return optimal_split_val(
            X[my_idxs], dim, possible_vals=possible_vals, X_orig=X_orig,
            return_possible_vals_losses=return_possible_vals_losses)

    def col_means(self):
        return self.sumX.astype(np.float64) / max(1, self.N)

    def col_variances(self, safe=False):
        if self.N < 1:
            return np.zeros(self.D, dtype=np.float32)
        E_X2 = self.sumX2 / self.N
        E_X = self.sumX / self.N
        ret = E_X2 - (E_X * E_X)
        return np.maximum(0, ret) if safe else ret

    def col_sum_sqs(self):
        return self.col_variances() * self.N

    @property
    def loss(self):
        # if self.N < 1:
        #     return 0

        # # less stable version with one less divide and mul
        # return max(0, np.sum(self.sumX2 - (self.sumX * (self.sumX / self.N))))

        # more stable version, that also clamps variance at 0
        return max(0, np.sum(self.col_sum_sqs()))
        # expected_X = self.sumX / self.N
        # expected_X2 = self.sumX2 / self.N
        # return max(0, np.sum(expected_X2 - (expected_X * expected_X)) * self.N)


# @numba.jit(nopython=True)  # numpy cumsum in insanely slow
# def _cumsum_cols(X):
#     X = np.copy(X)
#     for i in range(1, X.shape[0]):
#         X[i] += X[i - 1]
#     return X


# numpy cumsum in insanely slow; also, having the nested loops is twice
# as fast as assigning rows (ie, X[i] += X[i-1])
@numba.njit(fastmath=True)
def _cumsum_cols(X):
    out = np.empty(X.shape, X.dtype)
    for j in range(X.shape[1]):
        out[0, j] = X[0, j]
    for i in range(1, X.shape[0]):
        for j in range(X.shape[1]):
            out[i, j] = X[i, j] + out[i - 1, j]
    return out


@numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
def _cumsse_cols(X):
    N, D = X.shape
    cumsses = np.empty((N, D), X.dtype)
    cumX_row = np.empty(D, X.dtype)
    cumX2_row = np.empty(D, X.dtype)
    for j in range(D):
        cumX_row[j] = X[0, j]
        cumX2_row[j] = X[0, j] * X[0, j]
        cumsses[0, j] = 0  # no err in bucket with 1 element
    for i in range(1, N):
        one_over_count = 1. / (i + 1)
        for j in range(D):
            cumX_row[j] += X[i, j]
            cumX2_row[j] += X[i, j] * X[i, j]
            meanX = cumX_row[j] * one_over_count
            cumsses[i, j] = cumX2_row[j] - (cumX_row[j] * meanX)
    return cumsses


# def optimal_split_val(X, dim, possible_vals=None, return_val_idx=False):
# @_memory.cache
def optimal_split_val(X, dim, possible_vals=None, X_orig=None,
                      # return_possible_vals_losses=False, force_val='median'):
                      return_possible_vals_losses=False, force_val=None,
                      # shrink_towards_median=True):
                      shrink_towards_median=False):

    X_orig = X if X_orig is None else X_orig
    # X_orig = X # TODO rm
    if X_orig.shape != X.shape:
        print("X orig shape: ", X_orig.shape)
        print("X shape: ", X.shape)
        assert X_orig.shape == X.shape

    if force_val in ('mean', 'median'):
        assert not return_possible_vals_losses
        x = X_orig[:, dim]
        val = np.median(x) if force_val == 'median' else np.mean(x)
        mask = X_orig < val
        X0 = X[mask]
        errs0 = X0 - X0.mean(axis=0)
        loss0 = np.sum(errs0 * errs0)
        X1 = X[~mask]
        errs = X1 - X1.mean(axis=0)
        loss1 = np.sum(errs * errs)
        return val, loss0 + loss1

    N, D = X.shape
    # sort_idxs = np.argsort(X[:, dim])
    sort_idxs = np.argsort(X_orig[:, dim])
    X_sort = X[sort_idxs]

    # use_jit = False
    use_jit = True
    if use_jit:
        # X_sort = X_sort[:100] # TODO rm
        # X_sort = np.ascontiguousarray(X_sort)
        # N, D = X_sort.shape
        # print("about to call jitted func; N, D = ", N, D)
        sses_head = _cumsse_cols(X_sort)
        # print("got thru first call...")
        # X_sort_rev = np.ascontiguousarray(X_sort[::-1])
        # sses_tail = _cumsse_cols(X_sort_rev)[::-1]
        sses_tail = _cumsse_cols(X_sort[::-1])[::-1]
        # print("returned from jitted func!")
    else:
        X_sort_sq = X_sort * X_sort
        # cumX_head = np.cumsum(X_sort, axis=0)
        # cumX2_head = np.cumsum(X_sort_sq, axis=0)
        # cumX_tail = np.cumsum(X_sort[::-1], axis=0)[::-1]
        # cumX2_tail = np.cumsum(X_sort_sq[::-1], axis=0)[::-1]
        cumX_head = _cumsum_cols(X_sort)
        cumX2_head = _cumsum_cols(X_sort_sq)
        cumX_tail = _cumsum_cols(X_sort[::-1])[::-1]
        cumX2_tail = _cumsum_cols(X_sort_sq[::-1])[::-1]

        all_counts = np.arange(1, N + 1).reshape(-1, 1)
        EX_head = cumX_head / all_counts            # E[X], starting from 0
        EX_tail = cumX_tail / all_counts[::-1]      # E[X], starting from N-1
        # EX2_head = cumX2_head / all_counts          # E[X^2], starting from 0
        # EX2_tail = cumX2_tail / all_counts[::-1]    # E[X^2], starting from N-1
        # mses_head = EX2_head - (EX_head * EX_head)  # mses from 0
        # mses_tail = EX2_tail - (EX_tail * EX_tail)  # mses from N-1
        # sses_head = mses_head * all_counts          #
        # sses_tail = mses_tail * all_counts[::-1]

        # simpler equivalent of above; mse * N reduces to this
        sses_head = cumX2_head - (cumX_head * EX_head)
        sses_tail = cumX2_tail - (cumX_tail * EX_tail)

    # # TODO rm
    # mse_head_diffs = sses_head[1:] - sses_head[:-1]
    # # print("mse_head_diffs[:20]", mse_head_diffs[:20])
    # assert np.all(mse_head_diffs > -.1)  # should be nondecreasing
    # mse_tail_diffs = sses_tail[1:] - sses_tail[:-1]
    # assert np.all(mse_tail_diffs < .1)  # should be nonincreasing

    sses = sses_head
    sses[:-1] += sses_tail[1:]  # sse of X_sort[:i] + sse of X_sort[i:]
    sses = sses.sum(axis=1)

    if shrink_towards_median:
        minsse, maxsse = np.min(sses), np.max(sses)
        scale = maxsse - minsse
        # n_over_2 = N // 2
        # scale = (maxsse - minsse) / n_over_2
        coeffs = np.abs(np.arange(N, dtype=np.float32))
        penalties = coeffs * (scale / np.max(coeffs))
        sses += penalties

    # # TODO rm
    # E_X = X.mean(axis=0)
    # E_X2 = (X * X).mean(axis=0)
    # sse_true = np.sum(E_X2 - (E_X * E_X)) * N
    # print("sses[0], sses[-1], true loss, np.sum(X.var(axis=0)) * N",
    #       sses[0], sses[-1], sse_true, np.sum(X.var(axis=0)) * N)

    # X_orig_sort = X_orig[sort_idxs]
    if possible_vals is None or not len(possible_vals):  # can split anywhere
        best_idx = np.argmin(sses)
        next_idx = min(N - 1, best_idx + 1)
        # best_val = (X_sort[best_idx, dim] + X_sort[next_idx, dim]) / 2.
        # X_orig_sort = X_orig[sort_idxs]
        col = X_orig[:, dim]
        best_val = (col[sort_idxs[best_idx]] + col[sort_idxs[next_idx]]) / 2
        # best_val = (X_orig_sort[best_idx, dim] + X_orig_sort[next_idx, dim]) / 2
    else:  # have to choose one of the values in possible_vals
        sorted_col = X_orig[:, dim][sort_idxs]
        idxs = np.searchsorted(sorted_col, possible_vals)
        # idxs = np.unique(idxs)
        idxs = np.maximum(0, idxs - 1)  # searchsorted returns first idx larger
        sses_for_idxs = sses[idxs]
        which_idx_idx = np.argmin(sses_for_idxs)
        best_idx = idxs[which_idx_idx]
        best_val = possible_vals[which_idx_idx]

    # print("return_possible_vals_losses: ", return_possible_vals_losses)
    ret = best_val, sses[best_idx]
    return ret + (sses_for_idxs,) if return_possible_vals_losses else ret


def evenly_spaced_quantiles(x, nquantiles, dedup=True):
    x = np.unique(x)

    # handle x with fewer unique elements than nquantiles (or same number, or
    # not that many more; basically just want each returned value to be uniq
    # and useful for binning the distribution)
    if len(x) == nquantiles:
        return x
    elif len(x) == 1:
        return np.linspace(-1, 3, num=nquantiles) * x[0]
    elif len(x) < 2 * nquantiles:
        return np.linspace(np.min(x), np.max(x), num=nquantiles)

    n = nquantiles + 1
    fracs = np.arange(1, n) / float(n)
    return np.array([np.quantile(x, frac) for frac in fracs])


class PointInfo(object):
    __slots__ = 'data bucket_id'.split()

    def __init__(self, data, bucket_id):
        self.data = data
        self.bucket_id = bucket_id


class Split(object):
    __slots__ = 'dim val loss_change'.split()

    def __init__(self, dim, val, loss_change=None):
        self.dim = dim
        self.val = val
        self.loss_change = loss_change


def _sort_and_append_orig_idx(x, ascending=True):
    sort_idxs = np.argsort(x)
    if not ascending:
        sort_idxs = sort_idxs[::-1]
    x_sort = x[sort_idxs]
    orig_idxs = np.arange(len(x))[sort_idxs]
    return list(zip(x_sort, orig_idxs))


def _split_existing_buckets(buckets):
    return [buck.split() for buck in buckets]
    # new_buckets = []
    # # D = len(buckets[0].sumX)
    # for buck in buckets:
    #     # buck0 = copy.deepcopy(bucket)
    #     # buck0 = Bucket(N=buck.N, D=D, point_ids=copy.deepcopy(buck.point_ids),
    #     #                sumX=np.copy(buck.sumX), sumX2=np.copy(buck.sumX2))
    #     # buck0 = buck.copy()
    #     # buck1 = Bucket(D=buckets[0].D)
    #     new_buckets.append((buck0, buck1))
    # return new_buckets


class MultiSplit(object):
    __slots__ = 'dim vals scaleby offset'.split()

    def __init__(self, dim, vals, scaleby=None, offset=None):
        self.dim = dim
        self.vals = np.asarray(vals)
        self.scaleby = scaleby
        self.offset = offset

    def preprocess_x(self, x):
        if self.offset is not None:
            x = x - self.offset
        if self.scaleby is not None:
            x = x * self.scaleby
        return x


def learn_multisplits_orig(X, nsplits, log2_max_vals_per_split=4,
                      try_nquantiles=16, return_centroids=True,
                      # learn_quantize_params=False,
                      learn_quantize_params='int16',
                      # learn_quantize_params=True,
                      # verbose=1):
                      verbose=2):
                      # verbose=3):
    X = X.astype(np.float32)
    N, D = X.shape
    max_vals_per_split = 1 << log2_max_vals_per_split

    X_hat = np.zeros_like(X)

    # initially, one big bucket with everything
    buckets = [Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0),
               point_ids=np.arange(N))]
    total_loss = sum([bucket.loss for bucket in buckets])

    # values to try in each dim, after buckets no longer get to pick optimal
    # ones; we try values that at evenly spaced quantiles
    possible_split_vals = np.empty((D, try_nquantiles), dtype=X.dtype)
    for dim in range(D):
        # possible_split_vals[dim] = evenly_spaced_quantiles(
        #     X[:, dim], try_nquantiles)

        # exclude enpoints, so we get appropriate number of points linearly
        # spaced *between* min and max values
        minval, maxval = np.min(X[:, dim]), np.max(X[:, dim])
        possible_split_vals[dim] = np.linspace(
            minval, maxval, num=(try_nquantiles + 2))[1:-1]

    # print("initial possible split vals: ")
    # print(possible_split_vals[:8])
    # print(possible_split_vals[8:16])
    # import sys; sys.exit()

    if verbose > 0:
        print("================================")
        print("learn_multisplits(): initial loss: ", total_loss)

    splits = []
    col_losses = np.zeros(D, dtype=np.float32)  # TODO rm?
    for s in range(nsplits):
        # if s >= 2:
        #     print("exiting after two splits")
        #     import sys; sys.exit()
        if verbose > 1:
            print("------------------------ finding split #:", s)
        for i, buck in enumerate(buckets):  # TODO rm sanity check
            assert buck.id == i

        nbuckets = len(buckets)

        # compute number of bucket groups and size of each
        ngroups = min(nbuckets, max_vals_per_split)
        nbuckets_per_group = nbuckets // ngroups
        assert nbuckets_per_group * ngroups == nbuckets  # sanity check

        # try_ndims = 8
        # try_ndims = 4
        try_ndims = 1
        # dim_heuristic = 'eigenvec'
        # dim_heuristic = 'bucket_eigenvecs'
        dim_heuristic = 'variance'
        if dim_heuristic == 'eigenvec':
            # compute current reconstruction of X, along with errs
            for buck in buckets:
                # print("point ids: ", buck.point_ids)
                if len(buck.point_ids):
                    centroid = buck.col_means()
                    # X_hat[np.array(buck.point_ids)] = centroid
                    X_hat[buck.point_ids] = centroid
            X_res = X - X_hat
            # pick dims by looking at top principal component
            v = subs.top_principal_component(X_res)
            try_dims = np.argsort(np.abs(v))[-try_ndims:]
        elif dim_heuristic == 'bucket_eigenvecs':
            dim_scores = np.zeros(D, dtype=np.float32)
            for buck in buckets:
                if buck.N < 2:
                    continue
                X_buck = X[buck.point_ids]
                v, lamda = subs.top_principal_component(
                    X_buck, return_eigenval=True)
                v *= lamda
                dim_scores += np.abs(v)
                # X_buck -= X_buck.mean(axis=0)
            try_dims = np.argsort(dim_scores)[-try_ndims:]
        elif dim_heuristic == 'variance':
            # pick out dims to consider splitting on
            # try_dims = np.arange(D)  # TODO restrict to subset?
            col_losses[:] = 0
            for buck in buckets:
                col_losses += buck.col_sum_sqs()
            # try_dims = np.argsort(col_losses)[-8:]
            try_dims = np.argsort(col_losses)[-try_ndims:]
            # try_dims = np.argsort(col_losses)[-2:]
            # try_dims = np.arange(2)
            # try_dims = np.arange(D)  # TODO restrict to subset?

        losses = np.zeros(len(try_dims), dtype=X.dtype)
        losses_for_vals = np.zeros(try_nquantiles, dtype=X.dtype)
        all_split_vals = []  # vals chosen by each bucket/group for each dim

        # determine for this dim what the best split vals are for each
        # group and what the loss is when using these split vals
        for d, dim in enumerate(try_dims):
            if verbose > 2:
                # print("---------------------- dim = ", dim)
                print("======== dim = {}, ({:.5f}, {:.5f})".format(
                    dim, np.min(X[:, dim]), np.max(X[:, dim])))
            # just let each bucket pick its optimal split val for this dim;
            # special case of below where each "group" is one bucket, and
            # instead of having to pick val from fixed set, can be anything
            if nbuckets_per_group == 1:
                split_vals = []  # each bucket contributes one split val
                for buck in buckets:
                    val, loss = buck.optimal_split_val(X, dim)
                    losses[d] += loss
                    split_vals.append(val)
                all_split_vals.append(split_vals)
            # buckets have to pick from fixed set of possible values; each
            # group of buckets (defined by common prefix) have to agree on
            # one val, so we sum loss for each possible value across all
            # buckets in the group, and then take val with lowest sum
            else:
                split_vals = []  # each group contributes one split val

                for g in range(ngroups):
                    # print("------------------------ group #", g)
                    start_idx = g * nbuckets_per_group
                    end_idx = start_idx + nbuckets_per_group
                    group_buckets = buckets[start_idx:end_idx]
                    # print("bucket ids, counts: ",
                    #       [buck.id for buck in group_buckets],
                    #       [buck.N for buck in group_buckets])

                    # compute loss for each possible split value, summed
                    # across all buckets in this group; then choose best
                    possible_vals = possible_split_vals[dim]
                    # print("possible split vals: ", possible_vals)
                    losses_for_vals[:] = 0
                    # losses_for_vals = np.zeros_like(losses_for_vals)
                    # print("losses for vals: ", losses_for_vals)
                    for b, buck in enumerate(group_buckets):
                        _, _, val_losses = buck.optimal_split_val(
                            X, dim, possible_vals=possible_vals,
                            return_possible_vals_losses=True)
                        losses_for_vals += val_losses
                    best_val_idx = np.argmin(losses_for_vals)
                    best_val = possible_vals[best_val_idx]
                    best_loss = losses_for_vals[best_val_idx]
                    losses[d] += best_loss
                    # print("best {val idx, val, loss} = ",
                    #       best_val_idx, best_val, best_loss)
                    split_vals.append(best_val)
                all_split_vals.append(split_vals)

        # determine best dim to split on, and pull out associated split
        # vals for all buckets
        best_tried_dim_idx = np.argmin(losses)
        best_dim = try_dims[best_tried_dim_idx]
        use_split_vals = all_split_vals[best_tried_dim_idx]
        split = MultiSplit(dim=best_dim, vals=use_split_vals)
        if learn_quantize_params:
            # if len(use_split_vals) > 1:  # after 1st split
            #     minsplitval = np.min(use_split_vals)
            #     maxsplitval = np.max(use_split_vals)
            #     gap = maxsplitval - minsplitval
            #     offset = minsplitval - .02 * gap
            #     scale = 250. / gap  # slightly below 255. / gap
            # else:  # 1st split; only one bucket, so no intersplit range
            #     assert np.min(use_split_vals) == np.max(use_split_vals)
            #     x = X[:, best_dim]
            #     offset = np.min(x)
            #     scale = 255. / np.max(x - offset)
            #     # x -= offset
            #     # scale = 128. / np.max(split.vals - offset)
            #     # scale = 1 # TODO rm

            # # x = X[:, best_dim].copy()
            # x = X[:, best_dim]
            # offset = np.min(x)
            # # scale = 255. / np.max(x - offset)
            # scale = 250. / np.max(use_split_vals)  # slightly below 255

            # simple version, which also handles 1 bucket: just set min
            # value to be avg of min splitval and xval, and max value to
            # be avg of max splitval and xval
            x = X[:, best_dim]
            offset = (np.min(x) + np.min(use_split_vals)) / 2
            upper_val = (np.max(x) + np.max(use_split_vals)) / 2 - offset
            scale = 254. / upper_val
            if learn_quantize_params == 'int16':
                scale = 2. ** int(np.log2(scale))

            split.offset = offset
            split.scaleby = scale
            split.vals = (split.vals - split.offset) * split.scaleby
            split.vals = np.clip(split.vals, 0, 255).astype(np.int32)

        splits.append(split)

        # apply this split to get next round of buckets
        new_buckets = []
        for i, buck in enumerate(buckets):
            group_idx = i // nbuckets_per_group
            val = use_split_vals[group_idx]
            new_buckets += list(buck.split(X, dim=best_dim, val=val))
        buckets = new_buckets

        if verbose > 1:
            print("bucket counts: ", [buck.N for buck in buckets])
            print("loss from buckets: ",
                  sum([bucket.loss for bucket in buckets]))
            print("dim losses: ", losses)
            if verbose > 2:
                print("loss from sse computation: ",
                      losses[best_tried_dim_idx])
                print("using dim, split_vals:", best_dim, use_split_vals)

    # maybe return centroids in addition to set of MultiSplits and loss
    loss = sum([bucket.loss for bucket in buckets])
    if verbose > 0:
        print("learn_multisplits(): returning loss: ", loss)
    if return_centroids:
        centroids = np.vstack([buck.col_means() for buck in buckets])
        assert centroids.shape == (len(buckets), X.shape[1])
        return splits, loss, centroids
    return splits, loss


@_memory.cache
def learn_multisplits(
        X, nsplits=4, return_centroids=True, return_buckets=False,
        # learn_quantize_params=False,
        # learn_quantize_params='int16', X_orig=None, try_ndims=1,
        # learn_quantize_params='int16', X_orig=None, try_ndims=2,
        learn_quantize_params='int16', X_orig=None, try_ndims=4,
        # learn_quantize_params='int16', X_orig=None, try_ndims=8,
        # learn_quantize_params='int16', X_orig=None, try_ndims=16,
        # learn_quantize_params=True,
        # verbose=3):
        # verbose=2):
        verbose=1):
    assert nsplits <= 4  # >4 splits means >16 split_vals for this func's impl

    X = X.astype(np.float32)
    N, D = X.shape
    X_orig = X if X_orig is None else X_orig

    X_hat = np.zeros_like(X)

    # initially, one big bucket with everything
    buckets = [Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0),
               point_ids=np.arange(N))]
    total_loss = sum([bucket.loss for bucket in buckets])

    if verbose > 0:
        print("================================")
        # print("learn_multisplits(): initial loss: ", total_loss)
        print("learn_multisplits(): initial loss:   ", total_loss)
        # print("learn_multisplits(): trying ndims:   ", min(D, try_ndims))

    splits = []
    col_losses = np.zeros(D, dtype=np.float32)  # TODO rm?
    for s in range(nsplits):
        if verbose > 1:
            print("------------------------ finding split #:", s)

        # dim_heuristic = 'eigenvec'
        # dim_heuristic = 'bucket_eigenvecs'
        dim_heuristic = 'bucket_sse'
        # dim_heuristic = 'kurtosis'
        if dim_heuristic == 'eigenvec':
            # compute current reconstruction of X, along with errs
            if s > 0:
                for buck in buckets:
                    # print("point ids: ", buck.point_ids)
                    if len(buck.point_ids):
                        centroid = buck.col_means()
                        # X_hat[np.array(buck.point_ids)] = centroid
                        X_hat[buck.point_ids] = centroid
                X_res = X - X_hat
            else:
                X_res = X
            # pick dims by looking at top principal component
            v = subs.top_principal_component(X_res)
            try_dims = np.argsort(np.abs(v))[-try_ndims:]
        elif dim_heuristic == 'bucket_eigenvecs':
            dim_scores = np.zeros(D, dtype=np.float32)
            for buck in buckets:
                if buck.N < 2:
                    continue
                X_buck = X[buck.point_ids]
                v, lamda = subs.top_principal_component(
                    X_buck, return_eigenval=True)
                v *= lamda
                dim_scores += np.abs(v)
                # X_buck -= X_buck.mean(axis=0)
            try_dims = np.argsort(dim_scores)[-try_ndims:]
        elif dim_heuristic == 'bucket_sse':
            col_losses[:] = 0
            for buck in buckets:
                col_losses += buck.col_sum_sqs()
            try_dims = np.argsort(col_losses)[-try_ndims:]
        elif dim_heuristic == 'kurtosis':
            # compute X_res
            if s > 0:
                for buck in buckets:
                    # print("point ids: ", buck.point_ids)
                    if len(buck.point_ids):
                        centroid = buck.col_means()
                        # X_hat[np.array(buck.point_ids)] = centroid
                        X_hat[buck.point_ids] = centroid
                X_res = X - X_hat
            else:
                X_res = X

            col_losses[:] = 0
            for buck in buckets:
                col_losses += buck.col_sum_sqs()
            try_dims = np.argsort(col_losses)[-try_ndims:]

            from scipy import stats
            col_losses *= col_losses  # just 4th central moment
            col_losses *= stats.kurtosis(X_res, axis=0)
            try_dims = np.argsort(col_losses)[-try_ndims:]

        losses = np.zeros(len(try_dims), dtype=X.dtype)
        all_split_vals = []  # vals chosen by each bucket/group for each dim

        # determine for this dim what the best split vals are for each
        # group and what the loss is when using these split vals
        # print("try_dims: ", try_dims)
        for d, dim in enumerate(try_dims):
            # print("s, d, dim = ", s, d, dim)
            if verbose > 2:
                # print("---------------------- dim = ", dim)
                print("======== dim = {}, ({:.5f}, {:.5f})".format(
                    dim, np.min(X[:, dim]), np.max(X[:, dim])))
            split_vals = []  # each bucket contributes one split val
            for b, buck in enumerate(buckets):
                val, loss = buck.optimal_split_val(X, dim, X_orig=X_orig)
                losses[d] += loss
                if d > 0 and losses[d] >= np.min(losses[:d]):
                    if verbose > 2:
                        print("early abandoning after bucket {}!".format(b))
                    break  # this dim already can't be the best
                split_vals.append(val)
            all_split_vals.append(split_vals)

        # determine best dim to split on, and pull out associated split
        # vals for all buckets
        best_tried_dim_idx = np.argmin(losses)
        best_dim = try_dims[best_tried_dim_idx]
        use_split_vals = all_split_vals[best_tried_dim_idx]
        split = MultiSplit(dim=best_dim, vals=use_split_vals)
        if learn_quantize_params:
            # simple version, which also handles 1 bucket: just set min
            # value to be avg of min splitval and xval, and max value to
            # be avg of max splitval and xval
            x = X[:, best_dim]
            offset = (np.min(x) + np.min(use_split_vals)) / 2
            upper_val = (np.max(x) + np.max(use_split_vals)) / 2 - offset
            scale = 254. / upper_val
            if learn_quantize_params == 'int16':
                scale = 2. ** int(np.log2(scale))

            split.offset = offset
            split.scaleby = scale
            split.vals = (split.vals - split.offset) * split.scaleby
            split.vals = np.clip(split.vals, 0, 255).astype(np.int32)

        splits.append(split)

        # apply this split to get next round of buckets
        new_buckets = []
        for i, buck in enumerate(buckets):
            group_idx = i
            val = use_split_vals[group_idx]
            new_buckets += list(buck.split(X, dim=best_dim, val=val,
                                X_orig=X_orig))
        buckets = new_buckets

        if verbose > 1:
            print("bucket counts: ", [buck.N for buck in buckets])
            # print("loss from buckets: ",
            #       sum([bucket.loss for bucket in buckets]))
            print("dim losses: ", losses)
            if verbose > 2:
                print("loss from sse computation: ",
                      losses[best_tried_dim_idx])
                print("using dim, split_vals:", best_dim, use_split_vals)

    # maybe return centroids in addition to set of MultiSplits and loss
    loss = sum([bucket.loss for bucket in buckets])
    if verbose > 0:
        print("learn_multisplits(): returning loss: ", loss)

    ret = [splits, loss]
    if return_centroids:
        centroids = np.vstack([buck.col_means() for buck in buckets])
        assert centroids.shape == (len(buckets), X.shape[1])
        ret.append(centroids)
        # return splits, loss, centroids
    if return_buckets:
        # print("returning buckets!")
        ret.append(buckets)
    return tuple(ret)


@numba.njit(fastmath=True, cache=True)
def _XtX_encoded(X_enc, K=16):
    N, C = X_enc.shape
    D = C * K  # note that this is total number of centroids, not orig D

    out = np.zeros((D, D), np.int32)
    # out = np.zeros((D, D), np.float32)
    # D = int(C * K)  # note that this is total number of centroids, not orig D
    # out = np.zeros((D, D), np.int8)

    for n in range(N):
        for c in range(C):
            code_left = X_enc[n, c]
            dim_left = (K * c) + code_left
            out[dim_left, dim_left] += 1
            for cc in range(c + 1, C):
                code_right = X_enc[n, cc]
                dim_right = (K * cc) + code_right
                out[dim_left, dim_right] += 1

    # populate lower triangle
    for d in range(D):
        for dd in range(d + 1, D):
            out[dd, d] = out[d, dd]

    return out


@numba.njit(fastmath=True, cache=True)
def _XtY_encoded(X_enc, Y, K=16):
    N, C = X_enc.shape
    N, M = Y.shape

    D = int(C * K)  # note that this is total number of centroids, not orig D
    out = np.zeros((D, M), Y.dtype)

    for n in range(N):
        for c in range(C):
            code_left = X_enc[n, c]
            dim_left = (K * c) + code_left
            for m in range(M):
                out[dim_left, m] += Y[n, m]

    return out


@numba.njit(fastmath=True, cache=True)
def _XW_encoded(X_enc, W, K=16):
    N, C = X_enc.shape
    D, M = W.shape

    out = np.zeros((N, M), W.dtype)

    for n in range(N):
        for c in range(C):
            code_left = X_enc[n, c]
            dim_left = (K * c) + code_left
            for m in range(M):
                out[n, m] += W[dim_left, m]

    return out


@numba.njit(fastmath=True, cache=True)
def _densify_X_enc(X_enc, K=16):
    N, C = X_enc.shape
    D = C * K
    out = np.zeros((N, D), np.int8)
    for n in range(N):
        for c in range(C):
            code_left = X_enc[n, c]
            dim_left = (K * c) + code_left
            out[n, dim_left] = 1

    return out


def _fit_ridge_enc(X_enc=None, Y=None, K=16, lamda=1, X_bin=None):
    if X_bin is None:
        X_bin = _densify_X_enc(X_enc, K=K)
    est = linear_model.ridge.Ridge(fit_intercept=False, alpha=lamda)
    est.fit(X_bin, Y)
    return est.coef_.T


def encoded_lstsq(X_enc=None, X_bin=None, Y=None, K=16, XtX=None, XtY=None,
                  precondition=True, stable_ridge=True):

    if stable_ridge:
        return _fit_ridge_enc(X_enc=X_enc, Y=Y, X_bin=X_bin, K=K, lamda=1)

    if XtX is None:
        XtX = _XtX_encoded(X_enc, K=K).astype(np.float32)
        lamda = 1  # TODO cross-validate to get lamda

        # N = X_enc.shape[0]
        # # lamda = N / (K * K)
        # Y_bar = Y - Y.mean(axis=0)
        # lamda = N * np.var(Y - Y.mean(axis=0)) / (K * K)
        # # lamda = N * np.var(Y - Y.mean(axis=0)) / K
        # lamda = N * np.var(Y) / K
        # lamda = N * np.var(Y) / (K * K)
        # # lamda = N * 1e4  # should shrink coeffs to almost 0
        # # alpha = unscaled_alpha * np.var(X - X.mean(axis=0)) * N / D
        # lamda = N / (1e5)  # sorta works
        # lamda = N / (1e4) # sorta works

        lamda = max(1, lamda)
        print("using lamda = ", lamda)

        # lamda = max(1, len(X_enc) / 1e6)
        # lamda = max(1, len(X_enc) / 1e5)
        # lamda = max(1, len(X_enc) / 1e4)
        # lamda = max(1, len(X_enc) / float(K * K))
        # lamda = len(X_enc) / float(K)
        # print("computing and regularizing XtX using lambda = ", lamda)
        XtX += np.diag(np.ones(XtX.shape[0]) * lamda).astype(np.float32)  # ridge

    if XtY is None:
        XtY = _XtY_encoded(X_enc, Y, K=K)

    XtX = XtX.astype(np.float64)
    XtY = XtY.astype(np.float64)

    # preconditioning to avoid numerical issues (seemingly unnecessary, but
    # might as well do it)
    # scale = 1. / np.std(XtX)
    if precondition:

        # # pretend cols of X were scaled differently
        # xscales = np.linalg.norm(XtX, axis=0) + 1e-20
        # mulby = (1. / xscales)
        # XtX *= mulby * mulby
        # XtY *= mulby.reshape(-1, 1)

        # yscales = np.linalg.norm(XtY, axis=1) + 1e-20
        # yscales = np.linalg.norm(XtY, axis=0) + 1e-20
        # yscales = yscales.reshape(-1, 1)

        # xscales = np.mean(np.linalg.norm(XtX, axis=0))
        # xscales = 7
        # xscales = 1

        # XtY *= (1. / yscales)
        # XtY *= (1. / yscales.reshape(-1, 1))

        # scale = 1. / len(X_enc)
        scale = 1. / np.linalg.norm(XtX, axis=0).max()
        XtX = XtX * scale
        XtY = XtY * scale

    # W = np.linalg.solve(XtX, XtY)
    W, _, _, _ = np.linalg.lstsq(XtX, XtY, rcond=None) # doesn't fix it


    # W, _, _, _ = np.linalg.lstsq(X_bin, Y, rcond=None)


    # import torch
    # import torch.nn.functional as F
    # import torch.optim as optim

    # def _to_np(A):
    #     return A.cpu().detach().numpy()

    # niters = 10
    # for it in range(niters):


    # if precondition:
    #     pass
    #     # W *= xscales
    #     # W *= xscales.reshape(-1, 1)
    #     # W /= xscales.reshape(-1, 1)
    #     # W *= yscales.ravel()
    #     # W *= yscales

    # W *= yscales  # undo preconditioning

    # import matplotlib.pyplot as plt
    # _, axes = plt.subplots(2, 2, figsize=(13, 10))
    # axes[0, 0].imshow(_densify_X_enc(X_enc[:1000]), interpolation='nearest')
    # axes[0, 1].imshow(XtX, interpolation='nearest')
    # axes[1, 0].imshow(XtY, interpolation='nearest', cmap='RdBu')
    # axes[1, 1].imshow(W, interpolation='nearest', cmap='RdBu')
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    # import sys; sys.exit()

    return W


def _sparse_encoded_lstsq_gomp(X_enc, Y, nnz_blocks, K=16):
    assert nnz_blocks >= 1
    ncodebooks = X_enc.shape[1]
    M = Y.shape[1]

    # precompute XtX and XtY and create initial dense W
    XtX = _XtX_encoded(X_enc, K=K).astype(np.float32)
    XtX += np.diag(np.ones(XtX.shape[0])).astype(np.float32)  # ridge
    XtY = _XtY_encoded(X_enc, Y, K=K)
    W = encoded_lstsq(X_enc, Y, XtX=XtX, XtY=XtY)

    XtX = np.asfarray(XtX)  # since we'll be slicing columns

    keep_codebook_idxs = np.empty((M, nnz_blocks), dtype=np.int)

    XtX_G = np.empty((ncodebooks, K * ncodebooks, K), dtype=np.float32)
    for c in range(ncodebooks):
        start_idx = c * K
        end_idx = start_idx + K
        # use_XtX = XtX[start_idx:end_idx][:, start_idx:end_idx]
        use_XtX = XtX[:, start_idx:end_idx]
        XtX_G[c], _ = np.linalg.qr(use_XtX)  # KC x K

    codebook_scores = np.zeros(ncodebooks)
    for m in range(M):  # fully solve one output col at a time
        # xty = np.ascontiguousarray(XtY[:, m])
        targets = np.copy(XtY[:, m])
        residuals = targets
        keep_codebooks = set()
        w = np.copy(W[:, m])
        pq_codebook_idx = int(m / float(M) * ncodebooks)

        # print("---- m = ", m)
        for b in range(nnz_blocks):
            # targets_normed = targets

            # score each codebook to pick new one to add
            if b > 0:
                for c in range(ncodebooks):
                    if c in keep_codebooks:
                        codebook_scores[c] = -np.inf
                        continue
                    X_G = XtX_G[c]
                    codebook_scores[c] = np.linalg.norm(X_G.T @ residuals)
                keep_codebooks.add(np.argmax(codebook_scores))
            else:
                keep_codebooks.add(pq_codebook_idx)  # seed with pq idx
            # refit model using all the groups selected so far
            keep_idxs = [np.arange(i * K, (i + 1) * K)
                         for i in sorted(list(keep_codebooks))]
            keep_idxs = np.hstack(keep_idxs)
            XtX_subs = XtX[keep_idxs][:, keep_idxs]
            targets_subs = targets[keep_idxs]
            w_subs = np.linalg.solve(XtX_subs, targets_subs)
            # XtX_subs = XtX[:, keep_idxs]
            # targets_subs = targets[keep_idxs]
            # w_subs = np.linalg.solve(XtX_subs, targets)
            # w_subs, resid, _, _ = np.linalg.lstsq(XtX_subs, targets)
            w[:] = 0
            w[keep_idxs] = w_subs

            # resid_norm_sq = np.linalg.norm(residuals)**2
            # print("resid norm sq:     ", resid_norm_sq)
            # print("lstsq mse:         ", resid / resid_norm_sq)

            # residuals = targets - (XtX_subs @ w_subs)
            residuals = targets - (XtX[:, keep_idxs] @ w_subs)

            # resid_norm_sq = np.linalg.norm(residuals)**2
            # print("new resid norm sq: ", resid_norm_sq)

            # targets = np.copy(XtY[:, m]) - (XtX @ w)

        # update return arrays
        keep_codebook_idxs[m] = np.array(list(keep_codebooks))
        W[:, m] = w

    return W, keep_codebook_idxs


# each codebook has const number of nonzero idxs
def _sparse_encoded_lstsq_elim_v2(X_enc, Y, nnz_per_centroid, K=16,
                                  # uniform_sparsity=False):  # never better
                                  uniform_sparsity=True, pq_perm_algo='start',
                                  stable_ridge=True):
    ncodebooks = X_enc.shape[1]
    M = Y.shape[1]
    nnz_per_centroid = min(M, int(nnz_per_centroid))
    nnz_per_centroid = max(1, nnz_per_centroid)
    assert nnz_per_centroid >= int(np.ceil(M / ncodebooks))
    assert nnz_per_centroid <= M

    X_bin = _densify_X_enc(X_enc, K=K)

    if not stable_ridge:
        # precompute XtX and XtY and create initial dense W
        XtX = _XtX_encoded(X_enc, K=K).astype(np.float32)

        lamda = 1
        # # alpha = unscaled_alpha * np.var(X - X.mean(axis=0)) * N / D
        # # lamda = np.sqrt(ncodebooks)
        # N = XtX.shape[0]
        # lamda = N / (K * K)
        # lamda = max(1, lamda)
        # print("using lamda = ", lamda)

        # lamda = max(1, len(X_enc) / 1e4)
        # lamda = max(1, len(X_enc) / float(K * K))
        XtX += np.diag(np.ones(XtX.shape[0]) * lamda).astype(np.float32)  # ridge
        # XtX += np.diag(np.ones(XtX.shape[0])).astype(np.float32)  # ridge
        XtY = _XtY_encoded(X_enc, Y, K=K)

        # scale = 1. / len(X_enc)
        scale = 1. / np.linalg.norm(XtX, axis=0).max()
        XtX = XtX * scale
        XtY = XtY * scale

        W = encoded_lstsq(X_bin=X_bin, Y=Y, XtX=XtX, XtY=XtY, precondition=False,
                          stable_ridge=stable_ridge)  # KC x M

        XtX = np.asfarray(XtX)  # since we'll be slicing columns
    else:  # stable_ridge is True
        W = encoded_lstsq(X_bin=X_bin, Y=Y, stable_ridge=stable_ridge)

    # score all blocks of W
    all_scores = np.empty((ncodebooks, M), dtype=np.float)  # C x M
    for c in range(ncodebooks):
        Xc = X_enc[:, c].reshape(-1, 1)
        start_idx = c * K
        end_idx = start_idx + K
        Wc = W[start_idx:end_idx]

        Yc = _XtY_encoded(Xc, Wc, K=K)  # N x M
        all_scores[c] = np.linalg.norm(Yc, axis=0)

    # pq_idxs = _pq_codebook_start_end_idxs(M, ncodebooks)
    pq_idxs = _pq_codebook_start_end_idxs(Y, ncodebooks, algo=pq_perm_algo)

    # now pick which cols to keep in each codebook
    keep_mask = np.zeros((ncodebooks, M), dtype=np.bool)
    # subvec_len = int(np.ceil(M / ncodebooks))
    for c in range(ncodebooks):
        # initialize with PQ
        start_idx, end_idx = pq_idxs[c]
        keep_mask[c, start_idx:end_idx] = 1

        subvec_len = end_idx - start_idx
        assert subvec_len >= 1
        keep_nidxs_extra = nnz_per_centroid - subvec_len
        scores = all_scores[c]
        scores[start_idx:end_idx] = 0

        if uniform_sparsity and keep_nidxs_extra > 0:
            # take as many other (best) nonzero idxs as we we're allowed to
            assert len(scores) >= keep_nidxs_extra
            best_idxs = np.argsort(scores)[-keep_nidxs_extra:]
            if len(best_idxs) != keep_nidxs_extra:
                print("len(best_idxs)", len(best_idxs))
                print("keep_nidxs_extra", keep_nidxs_extra)
                assert len(best_idxs) == keep_nidxs_extra
            keep_mask[c, best_idxs] = True

    if not uniform_sparsity:
        scores = all_scores.ravel()
        nkept_idxs = M  # number of nonzeros used already
        keep_nidxs_total = nnz_per_centroid * ncodebooks
        keep_nidxs_extra = keep_nidxs_total - nkept_idxs
        keep_idxs = np.argsort(scores)[-keep_nidxs_extra:]
        flat_mask = keep_mask.ravel()
        flat_mask[keep_idxs] = 1
        keep_mask = flat_mask.reshape(keep_mask.shape)

    # at this point, we have the mask for which cols of each centroid to keep;
    # now we just need to go from a mask to a set of indices and a sparse
    # matrix of centroids
    W_sparse = np.empty((ncodebooks * K, M), dtype=np.float32)
    if uniform_sparsity:
        ret_idxs = np.empty((ncodebooks, nnz_per_centroid), dtype=np.int)
    else:
        ret_idxs = []
    # else:
        # ret_idxs = np.zeros((ncodebooks, M), dtype=np.int) - 1
    for c in range(ncodebooks):
        idxs = np.where(keep_mask[c] != 0)[0]
        if uniform_sparsity:
            if len(idxs) != nnz_per_centroid:
                print("c: ", c)
                print("len(idxs): ", len(idxs))
                print("nnz_per_centroid: ", nnz_per_centroid)
                print("keep_mask counts:", keep_mask.sum(axis=1))
                assert len(idxs) == nnz_per_centroid
            ret_idxs[c] = idxs
        else:
            ret_idxs.append(idxs)

        zero_idxs = np.where(keep_mask[c] == 0)[0]
        start_idx = c * K
        end_idx = start_idx + K
        Wc = W[start_idx:end_idx]
        Wc[:, zero_idxs] = 0
        W_sparse[start_idx:end_idx] = Wc

    # now refit W_sparse to each output col; right now it's just the original
    # W with a bunch of entries zeroed
    for m in range(M):
        w = W_sparse[:, m]
        keep_idxs = np.where(w != 0)[0]

        if stable_ridge:
            X_bin_subs = X_bin[:, keep_idxs]
            w_subs = _fit_ridge_enc(X_bin=X_bin_subs, Y=Y[:, m])
        else:
            xty = XtY[:, m]
            use_XtX = XtX[keep_idxs][:, keep_idxs]
            use_xty = xty[keep_idxs]
            w_subs = np.linalg.solve(use_XtX, use_xty)
        w[:] = 0
        w[keep_idxs] = w_subs
        W_sparse[:, m] = w

    # nnzs = [len(idxs) for idxs in ret_idxs]
    # print("nnzs: ", nnzs)

    # print(f"returning {ret_idxs.shape[1]} nonzeros per centroid...")
    return W_sparse, ret_idxs


def _sparse_encoded_lstsq_backward_elim(X_enc, Y, nnz_blocks, K=16):
    ncodebooks = X_enc.shape[1]
    eliminate_nblocks = ncodebooks - nnz_blocks
    M = Y.shape[1]

    # precompute XtX and XtY and create initial dense W
    XtX = _XtX_encoded(X_enc, K=K).astype(np.float32)
    XtX += np.diag(np.ones(XtX.shape[0])).astype(np.float32)  # ridge
    XtY = _XtY_encoded(X_enc, Y, K=K)
    W = encoded_lstsq(X_enc, Y, XtX=XtX, XtY=XtY)

    XtX = np.asfarray(XtX)  # since we'll be slicing columns

    keep_codebook_idxs = np.empty((M, nnz_blocks), dtype=np.int)

    codebook_scores = np.zeros(ncodebooks)
    for m in range(M):  # fully solve one output col at a time
        xty = np.ascontiguousarray(XtY[:, m])
        rm_codebook_idxs = set()
        w = np.copy(W[:, m])
        for b in range(eliminate_nblocks):
            # evaluate contribution of each codebook
            for c in range(ncodebooks):
                # if c in rm_codebook_idxs or c == pq_codebook_idx:
                if c in rm_codebook_idxs:
                    codebook_scores[c] = np.inf
                    continue

                start_idx = c * K
                end_idx = start_idx + K

                # XtX_subs = XtX[:, start_idx:end_idx]    # CK x K
                # w_subs = w[start_idx:end_idx]           # K
                # xtyhat_subs = XtX_subs @ w_subs         # CK x 1
                # codebook_scores[c] = np.linalg.norm(xtyhat_subs)

                XtX_subs = XtX[start_idx:end_idx][:, start_idx:end_idx]
                w_subs = w[start_idx:end_idx]           # K
                xtyhat_subs = XtX_subs @ w_subs         # K x 1
                codebook_scores[c] = np.linalg.norm(xtyhat_subs)

            # rm least helpful codebook and refit the least squares
            rm_codebook_idxs.add(np.argmin(codebook_scores))

            keep_codebooks = [i for i in range(ncodebooks)
                              if i not in rm_codebook_idxs]

            keep_idxs = [np.arange(i * K, (i + 1) * K)
                         for i in keep_codebooks]
            keep_idxs = np.hstack(keep_idxs)
            use_XtX = XtX[keep_idxs][:, keep_idxs]
            use_xty = xty[keep_idxs]
            w_subs = np.linalg.solve(use_XtX, use_xty)
            # print("w shape: ", w.shape)
            # print("rm codebooks: ", rm_codebook_idxs)
            # print("keep codebooks: ", keep_codebooks)
            # print("keep idxs: ", keep_idxs)
            # print("type(keep idxs): ", type(keep_idxs))
            # print("w[keep idxs]: ", w[keep_idxs])
            # print("resid: ", resid)
            w[:] = 0
            w[keep_idxs] = w_subs

        # update return arrays
        keep_idxs = [i for i in range(ncodebooks) if i not in rm_codebook_idxs]
        keep_codebook_idxs[m] = np.array(keep_codebooks)
        W[:, m] = w

    return W, keep_codebook_idxs  # CK x M, M x nnz


def sparse_encoded_lstsq(X_enc, Y, K=16, nnz_blocks=-1, **kwargs):
    ncodebooks = X_enc.shape[1]
    if nnz_blocks < 1:
        # nnz_per_centroid = Y.shape[1]
        # default to returning dense centroids
        W = encoded_lstsq(X_enc, Y, K=16)
        ncodebooks = X_enc.shape[1]
        M = Y.shape[1]
        keep_codebook_idxs = np.empty((ncodebooks, M), dtype=np.int)
        all_idxs = np.arange(M)
        for c in range(ncodebooks):
            keep_codebook_idxs[c] = all_idxs
        return W, keep_codebook_idxs
    else:
        nnz_per_centroid = int(nnz_blocks * Y.shape[1] / ncodebooks)

        # nnz_blocks = int(np.sqrt(ncodebooks) + .5)

    # return _sparse_encoded_lstsq_backward_elim(
    #     X_enc, Y, nnz_blocks=nnz_blocks, K=K)
    # return _sparse_encoded_lstsq_gomp(X_enc, Y, nnz_blocks=nnz_blocks, K=K)

    # print("nnz_per_centroid: ", nnz_per_centroid)
    return _sparse_encoded_lstsq_elim_v2(
        X_enc, Y, nnz_per_centroid=nnz_per_centroid, K=K, **kwargs)


# def _pq_codebook_start_end_idxs(D, ncodebooks):
def _pq_codebook_start_end_idxs(X, ncodebooks, algo='start'):
    assert algo in ('start', 'end')  # TODO do something smarter here

    # D = int(D)
    _, D = X.shape
    ncodebooks = int(ncodebooks)
    assert D >= ncodebooks

    idxs = np.empty((ncodebooks, 2), dtype=np.int)
    full_subvec_len = D // ncodebooks
    start_idx = 0
    for c in range(ncodebooks):
        subvec_len = full_subvec_len
        if algo == 'start':     # wider codebooks at the start
            if c < (D % ncodebooks):
                subvec_len += 1
        elif algo == 'end':     # wider codebooks at the end
            if (ncodebooks - c - 1) < (D % ncodebooks):
                subvec_len += 1
        end_idx = min(D, start_idx + subvec_len)
        # print("c, start_idx, end_idx: ", c, start_idx, end_idx)
        # print("start_idx, end_idx: ", c, start_idx, end_idx)
        idxs[c, 0] = start_idx
        idxs[c, 1] = end_idx

        start_idx = end_idx

    assert idxs[0, 0] == 0
    assert idxs[-1, -1] == D
    return idxs


@_memory.cache
def _learn_mithral_initialization(X, ncodebooks,
                                  pq_perm_algo='start', **kwargs):
    N, D = X.shape
    ncentroids_per_codebook = 16

    X = X.astype(np.float32)
    X_res = X.copy()
    X_orig = X

    all_centroids = np.zeros(
        (ncodebooks, ncentroids_per_codebook, D), dtype=np.float32)
    all_splits = []
    pq_idxs = _pq_codebook_start_end_idxs(X, ncodebooks, algo=pq_perm_algo)
    subvec_len = int(np.ceil(D / ncodebooks))  # for non-pq heuristics

    nonzeros_heuristic = 'pq'

    # ------------------------ 0th iteration; initialize all codebooks
    all_splits = []
    all_buckets = []
    for c in range(ncodebooks):
        if nonzeros_heuristic == 'pq':
            start_idx, end_idx = pq_idxs[c]
            idxs = np.arange(start_idx, end_idx)
        elif nonzeros_heuristic == 'pca':
            v = subs.top_principal_component(X_res)
            idxs = np.argsort(np.abs(v))[:-subvec_len]
        elif nonzeros_heuristic == 'disjoint_pca':
            use_X_res = X_res.copy()
            if c > 0:  # not the first codebook
                use_X_res[:, idxs] = 0  # can't use same subspace
            v = subs.top_principal_component(use_X_res)
            idxs = np.argsort(np.abs(v))[:-subvec_len]

        use_X_res = X_res[:, idxs]
        use_X_orig = X_orig[:, idxs]

        # learn codebook to soak current residuals
        multisplits, _, buckets = learn_multisplits(
            use_X_res, X_orig=use_X_orig,
            return_centroids=False, return_buckets=True, **kwargs)
        for split in multisplits:
            split.dim = idxs[split.dim]
        all_splits.append(multisplits)
        all_buckets.append(buckets)

        # update residuals and store centroids
        centroid = np.zeros(D, dtype=np.float32)
        for b, buck in enumerate(buckets):
            if len(buck.point_ids):
                centroid[:] = 0
                centroid[idxs] = buck.col_means()
                X_res[buck.point_ids] -= centroid
                # update centroid here in case we want to regularize it somehow
                all_centroids[c, b] = centroid

        # print("X_res mse / X mse: ",
        #       (X_res * X_res).mean() / (X_orig * X_orig).mean())

    return X_res, all_splits, all_centroids, all_buckets


@_memory.cache
def learn_mithral(X, ncodebooks, return_buckets=False,
                  lut_work_const=-1, **kwargs):
    N, D = X.shape
    ncentroids_per_codebook = 16
    X_orig = X.astype(np.float32)

    X_res0, all_splits0, all_centroids0, all_buckets0 = \
        _learn_mithral_initialization(X, ncodebooks, pq_perm_algo='start')

    mse_orig = (X_orig * X_orig).mean()
    mse0 = (X_res0 * X_res0).mean()
    print("X_res mse / X mse: ", mse0 / mse_orig)

    used_perm_algo = 'start'
    if False:
        # choose between having wider codebooks at the start vs the end (if
        # there might be a meaningful difference)
        X_res1, all_splits1, all_centroids1, all_buckets1 = \
            _learn_mithral_initialization(X, ncodebooks, pq_perm_algo='end')
        mse1 = (X_res1 * X_res1).mean()

        if mse0 <= mse1:
            X_res, all_splits, all_centroids, all_buckets = (
                X_res0, all_splits0, all_centroids0, all_buckets0)
        else:
            X_res, all_splits, all_centroids, all_buckets = (
                X_res1, all_splits1, all_centroids1, all_buckets1)
            used_perm_algo = 'end'

        print("X_res1 mse / X mse: ", mse1 / mse_orig)
    else:
        X_res, all_splits, all_centroids, all_buckets = (
            X_res0, all_splits0, all_centroids0, all_buckets0)

    # optimize centroids discriminatively conditioned on assignments
    X_enc = mithral_encode(X, all_splits)

    if lut_work_const != 1:  # if it's 1, equivalent to just doing PQ
        #
        # shrink W towards 0
        #
        # if lut_work_const < 0:
        #     W = encoded_lstsq(X_enc, X)
        # else:
        #     W, nonzero_blocks = sparse_encoded_lstsq(
        #         X_enc, X, nnz_blocks=lut_work_const)

        #
        # shrink W towards initial centroids
        #
        if lut_work_const < 0:
            print("fitting dense lstsq to X_res")
            W = encoded_lstsq(X_enc=X_enc, Y=X_res)
        else:
            W, _ = sparse_encoded_lstsq(
                    X_enc, X_res, nnz_blocks=lut_work_const,
                    pq_perm_algo=used_perm_algo)

        all_centroids_delta = W.reshape(ncodebooks, ncentroids_per_codebook, D)
        all_centroids += all_centroids_delta

        # check how much improvement we got
        X_res -= _XW_encoded(X_enc, W)  # if we fit to X_res
        mse_res = (X_res * X_res).mean()
        print("X_res mse / X mse after lstsq: ", mse_res / mse_orig)
        # print("min, median, max, std, of all centroids after lstsq:\n",
        #       all_centroids.min(), np.median(all_centroids),
        #       all_centroids.max(), all_centroids.std())

    if return_buckets:
        return all_splits, all_centroids, all_buckets
    return all_splits, all_centroids


def learn_mithral_v1(X, ncodebooks, niters=1, return_buckets=False, **kwargs):
    # print("called learn_mithral!"); import sys; sys.exit()
    N, D = X.shape
    ncentroids_per_codebook = 16

    X = X.astype(np.float32)
    X_res = X.copy()
    X_orig = X
    X_hat = np.zeros_like(X)

    all_centroids = np.zeros(
        (ncodebooks, ncentroids_per_codebook, D), dtype=np.float32)
    all_splits = []
    subvec_len = int(np.ceil(D / ncodebooks))
    # use_X_res = np.zeros_like(X_res)

    # TODO multiple iters; also store assignments from each codebook, so
    # that we can undo effect of its X_hat (can't store X_hat directly for
    # large data, so recompute on the fly using assignments and centroids)

    nonzeros_heuristic = 'pq'
    # nonzeros_heuristic = 'pca'
    # nonzeros_heuristic = 'disjoint_pca'

    # TODO store assignments (or maybe just buckets directly)
    # TODO update just centroids (not assignments) at iter end

    # ------------------------ 0th iteration; initialize all codebooks
    all_splits = []
    all_buckets = []
    for c in range(ncodebooks):
        if nonzeros_heuristic == 'pq':
            start_idx = c * subvec_len
            end_idx = min(D, start_idx + subvec_len)
            idxs = np.arange(start_idx, end_idx)
        elif nonzeros_heuristic == 'pca':
            v = subs.top_principal_component(X_res)
            idxs = np.argsort(np.abs(v))[:-subvec_len]
        elif nonzeros_heuristic == 'disjoint_pca':
            use_X_res = X_res.copy()
            if c > 0:  # not the first codebook
                use_X_res[:, idxs] = 0  # can't use same subspace
            v = subs.top_principal_component(use_X_res)
            idxs = np.argsort(np.abs(v))[:-subvec_len]

        use_X_res = X_res[:, idxs]
        use_X_orig = X_orig[:, idxs]

        # learn codebook to soak current residuals
        multisplits, _, buckets = learn_multisplits(
            use_X_res, X_orig=use_X_orig,
            return_centroids=False, return_buckets=True, **kwargs)
        for split in multisplits:
            split.dim = idxs[split.dim]
        all_splits.append(multisplits)
        all_buckets.append(buckets)

        # use_X_res[:, start_idx:end_idx] = 0
        # use_X_res[:] = 0

        # update residuals and store centroids
        centroid = np.zeros(D, dtype=np.float32)
        for b, buck in enumerate(buckets):
            if len(buck.point_ids):
                centroid[:] = 0
                centroid[idxs] = buck.col_means()
                # centroid /= 2 # TODO rm
                X_hat[buck.point_ids] = centroid
                # update centroid here in case we want to regularize it somehow
                all_centroids[c, b] = centroid
        X_res -= X_hat

        print("X res var / X var: ", X_res.var() / X_orig.var())

    # ------------------------ remaining iters
    for t in range(niters):
        # now update centroids given assignments and all other centroids
        # for _ in range(5):
        # for _ in range(20):
        for _ in range(10):
            for c in range(ncodebooks):
                # print("c: ", c)
                # undo effect of this codebook
                buckets = all_buckets[c]
                for b, buck in enumerate(buckets):
                    if len(buck.point_ids):
                        X_hat[buck.point_ids] = all_centroids[c, b]
                X_res += X_hat
                # update centroids based on residuals given all other codebooks
                for b, buck in enumerate(buckets):
                    if len(buck.point_ids):
                        centroid = X_res[buck.point_ids].mean(axis=0)

                        # keep_ndims = D // 2
                        # zero_idxs = np.argsort(np.abs(centroid))[:-keep_ndims]
                        # centroid[zero_idxs] = 0

                        # true_centroid = X_res[buck.point_ids].mean(axis=0)
                        # old_centroid = all_centroids[c, b]
                        # centroid = (true_centroid + old_centroid) / 2

                        X_hat[buck.point_ids] = centroid
                        all_centroids[c, b] = centroid
                X_res -= X_hat
            print("X res var / X var after centroid updates: ",
                  X_res.var() / X_orig.var())

        # now update assignments
        if t == niters - 1:
            break  # end after updating centroids, not assignments
        for c in range(ncodebooks):
            # print("c: ", c)
            # undo effect of this codebook
            buckets = all_buckets[c]
            # orig_loss = sum([buck.loss for buck in buckets])
            orig_loss = np.sum(X_res * X_res)
            for b, buck in enumerate(buckets):
                if len(buck.point_ids):
                    X_hat[buck.point_ids] = all_centroids[c, b]
            X_res += X_hat

            multisplits, loss, buckets = learn_multisplits(
                X_res, X_orig=X_orig,
                return_centroids=False, return_buckets=True, **kwargs)
            print("orig loss, loss: ", orig_loss, loss)
            if loss > orig_loss:
                X_res -= X_hat
                continue

            all_splits[c] = multisplits
            all_buckets[c] = buckets

            # update residuals and store centroids
            # centroid = np.zeros(D, dtype=np.float32)
            for b, buck in enumerate(buckets):
                if len(buck.point_ids):
                    centroid = buck.col_means()
                    # centroid /= 2 # TODO rm
                    X_hat[buck.point_ids] = centroid
                    # update centroid here in case we want to regularize it somehow
                    all_centroids[c, b] = centroid
            X_res -= X_hat

            print("new X res var / X var: ", X_res.var() / X_orig.var())

    if return_buckets:
        return all_splits, all_centroids, all_buckets
    return all_splits, all_centroids


def mithral_encode(X, multisplits_lists):
    N, D = X.shape
    ncodebooks = len(multisplits_lists)
    X_enc = np.empty((N, ncodebooks), dtype=np.int, order='f')
    for c in range(ncodebooks):
        X_enc[:, c] = assignments_from_multisplits(X, multisplits_lists[c])
    return np.ascontiguousarray(X_enc)


def mithral_lut(q, all_centroids):
    q = q.reshape(1, 1, -1)  # all_centroids is shape ncodebooks, ncentroids, D
    return (q * all_centroids).sum(axis=2)  # ncodebooks, ncentroids


def learn_splits_greedy(X, nsplits, verbose=2):
    N, D = X.shape
    assert nsplits <= D

    # # improve numerical stability
    # scale = np.std(X)
    # X *= (1. / scale)

    # precompute sorted lists of values within each dimension,
    # along with which row they originally were so look can look
    # up the whole vector (and bucket) associated with each value
    dim2sorted = []
    for dim in range(D):
        sorted_with_idx = _sort_and_append_orig_idx(X[:, dim])
        dim2sorted.append(sorted_with_idx)

    splits = []
    # buckets = [Bucket(N=N, sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0),
    buckets = [Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0),
               point_ids=np.arange(N))]

    # all_point_infos = [PointInfo(data=row, bucket_id=0) for row in X]
    bucket_assignments = np.zeros(N, dtype=np.int)

    # Z = X - X.mean(axis=0)
    # total_loss = np.sum(Z * Z)
    # print("initial SSE: ", total_loss)

    total_loss = sum([bucket.loss for bucket in buckets])
    if verbose > 0:
        print("learn_splits(): initial loss: ", total_loss)

    # unused_dims = set(np.arange(X.shape[1]))
    # all_dims = np.arange(D)

    col_losses = np.zeros(D, dtype=np.float32)  # TODO rm?

    for s in range(nsplits):
        if verbose > 1:
            print("================================ finding split #:", s)
        best_split = Split(dim=-1, val=-np.inf, loss_change=0)
        # for d in unused_dims:
        # for d in all_dims:
        # for d in all_dims[:2]:  # TODO rm

        col_losses[:] = 0
        for buck in buckets:
            col_losses += buck.col_sum_sqs()
        # try_dims = [np.argmax(col_losses)]
        # try_dims = np.argsort(col_losses)[-nsplits:]
        try_dims = np.argsort(col_losses)[-4:]
        # for d in [dim]:  # TODO multiple dim options?
        if verbose > 1:
            print("trying dims: ", try_dims)
            print("with losses: ", col_losses[try_dims])
        for d in try_dims:
            vals_and_point_ids = dim2sorted[d]
            new_buckets = _split_existing_buckets(buckets)
            new_total_loss = total_loss
            if verbose > 2:
                print("---------------------- dim = ", d)
            # for i, (val, point_id) in enumerate(vals_and_point_ids):

            # skip last point since that just puts everything in one bucket,
            # which is the same as what we're starting out with
            for val, point_id in vals_and_point_ids[:-1]:
                # if verbose > 1:
                #     print("i: {}/{}".format(i, len(vals_and_point_ids) - 1))

                # info = all_point_infos[point_id]
                # point, bucket_id = info.data, info.bucket_id
                point = X[point_id]
                bucket_id = bucket_assignments[point_id]

                bucket0 = new_buckets[bucket_id][0]
                bucket1 = new_buckets[bucket_id][1]
                old_loss = bucket0.loss + bucket1.loss
                bucket0.remove_point(point, point_id=point_id)
                bucket1.add_point(point, point_id=point_id)

                new_loss = bucket0.loss + bucket1.loss
                new_total_loss -= old_loss  # sub old loss from these buckets
                new_total_loss += new_loss  # add new loss from these buckets
                loss_change = new_total_loss - total_loss

                # if loss_change > .1:  # should be nonincreasing
                #     print("got loss change: ", loss_change)
                #     print("old total loss:", total_loss)
                #     print("new total loss:", new_total_loss)
                #     assert loss_change <= .1  # should be nonincreasing

                # # loss should be no worse than having new buckets unused
                # assert loss_change <= .1

                # if verbose > 2:
                #     print("-------- split point_id, val = ", point_id, val)
                #     print("bucket0 point ids, loss after update: ",
                #           bucket0.point_ids, bucket0.loss)
                #     print("bucket1 point ids, loss after update: ",
                #           bucket1.point_ids, bucket1.loss)
                #     print("loss change = {:.3f};\tnew_loss = {:.3f} ".format(
                #           loss_change, new_total_loss))

                if loss_change < best_split.loss_change:
                    best_split.dim = d
                    best_split.val = val
                    best_split.loss_change = loss_change

        if verbose > 2:
            print("---------------------- split on dim={}, val={:.3f} ".format(
                best_split.dim, best_split.val))

        buckets = [buck.split(X, dim=best_split.dim, val=best_split.val)
                   for buck in buckets]
        buckets = reduce(lambda b1, b2: b1 + b2, buckets)  # flatten pairs
        for i, buck in enumerate(buckets):
            ids = np.asarray(list(buck.point_ids), dtype=np.int)
            bucket_assignments[ids] = i

        total_loss = sum([bucket.loss for bucket in buckets])
        # unused_dims.remove(best_split.dim)
        splits.append(best_split)

        if verbose > 3:
            print('learn_splits(): new loss: {:.3f} from split at dim {}, '
                  'value {:.3f}'.format(
                    total_loss, best_split.dim, best_split.val))
            if verbose > 2:
                print('bucket losses: ')
                print([bucket.loss for bucket in buckets])
                print('bucket N, sumX, sumX2')
                print([bucket.N for bucket in buckets])
                print([list(bucket.sumX) for bucket in buckets])
                print([list(bucket.sumX2) for bucket in buckets])

    # for split in splits:
    #     split.val *= scale  # undo preconditioning
    # total_loss *= scale * scale

    return splits, total_loss


def learn_splits_conditional(X, nsplits, dim_algo='greedy_var',
                             split_algo='mean', **sink):
    N, D = X.shape
    assert nsplits <= D
    # unused_dims = set(np.arange(X.shape[1]))
    col_means = X.mean(axis=0)
    # dims = np.arange(D)
    used_mask = np.ones(D, dtype=np.float32)
    splits = []
    buckets = [Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0),
                      point_ids=np.arange(N))]
    col_losses = np.zeros(D, dtype=np.float32)
    for s in range(nsplits):
        print("---- learning split {}/{}...".format(s + 1, nsplits))
        print("current number of buckets: ", len(buckets))
        # col_vars = X.var(axis=0)
        col_losses[:] = 0
        for buck in buckets:
            col_losses += buck.col_sum_sqs()
        col_losses *= used_mask

        if dim_algo == 'greedy_var':
            dim = np.argmax(col_losses)
        used_mask[dim] = 0

        if split_algo == 'mean':
            val = col_means[dim]

        new_buckets = []
        for buck in buckets:
            new_buckets += list(buck.split(X=X, dim=dim, val=val))
        buckets = new_buckets

        splits.append(Split(dim=dim, val=val))

    return splits, -1


# def learn_splits_simple(X, nsplits, dim_algo='randunif', split_algo='mean',
# def learn_splits_simple(X, nsplits, dim_algo='greedy_var', split_algo='median',
def learn_splits_simple(X, nsplits, dim_algo='greedy_var', split_algo='mean',
                        **sink):
    # unused_dims = set(np.arange(X.shape[1]))
    unused_dims = list(np.arange(X.shape[1]))  # random.choice can't use set
    col_means = X.mean(axis=0)
    col_vars = X.var(axis=0)
    col_medians = np.median(X, axis=0)
    # overall_mean = np.mean(col_means)
    # overall_median = np.median(col_medians)
    # overall_var = X.var()

    var_idxs_descending = np.argsort(col_vars)[::-1]

    splits = []
    for s in range(nsplits):
        if dim_algo == 'randunif':
            dim = np.random.choice(unused_dims)
            unused_dims.remove(dim)
        elif dim_algo == 'greedy_var':
            dim = var_idxs_descending[s]

        if split_algo == 'mean':
            val = col_means[dim]
        elif split_algo == 'median':
            val = col_medians[dim]

        splits.append(Split(dim=dim, val=val))

    return splits, -1


def learn_splits(X, nsplits, return_centroids=True, algo='multisplits',
                 **kwargs):
    # indirect to particular func; will likely need to try something simpler
    # for debugging and/or as experimental control
    # return learn_splits_greedy(X, nsplits, **kwargs)
    # return learn_splits_simple(X, nsplits, **kwargs)
    # return learn_splits_conditional(X, nsplits, **kwargs)
    # return learn_splits_greedy(X, nsplits) # TODO fwd kwargs

    if algo == 'multisplits':
        return learn_multisplits(
            X, nsplits, return_centroids=return_centroids)

    if algo == 'splits':
        splits, loss = learn_splits_greedy(X, nsplits)

    if return_centroids:
        centroids = centroids_from_splits(X, splits)
        return splits, loss, centroids
    return splits, loss


def assignments_from_splits(X, splits):
    nsplits = len(splits)
    indicators = np.empty((nsplits, len(X)), dtype=np.int)
    for i, split in enumerate(splits):
        indicators[i] = X[:, split.dim] > split.val

    # compute assignments by treating indicators in a row as a binary num
    # scales = (2 ** np.arange(nsplits)).astype(np.int)
    scales = (1 << np.arange(nsplits)).astype(np.int)
    return (indicators.T * scales).sum(axis=1).astype(np.int)


def assignments_from_multisplits(X, splits):
    N, _ = X.shape
    nsplits = len(splits)
    # indicators = np.zeros((nsplits, len(X)), dtype=np.int)
    assert len(splits) >= 1
    # dim0 = splits[0].dim
    # assert len(splits[0].vals) == 1  # only 1 initial split
    # indicators[0] = X > splits[0].vals[0]

    max_ngroups = len(splits[-1].vals)
    nsplits_affecting_group_id = int(np.log2(max_ngroups))
    assert 1 << nsplits_affecting_group_id == max_ngroups  # power of 2
    # np.log2(max_nsplits)

    # determine group ids for each point; this is the one that's annoying
    # because the number of bits changes after split
    group_ids = np.zeros(N, dtype=np.int)
    for i in range(min(nsplits, nsplits_affecting_group_id)):
        split = splits[i]
        vals = split.vals[group_ids]
        # x = X[:, split.dim]
        # if split.offset is not None:
        #     x = x - split.offset
        # if split.scaleby is not None:
        #     x = x * split.scaleby
        # indicators = x > vals
        indicators = split.preprocess_x(X[:, split.dim]) > vals
        group_ids = (group_ids * 2) + indicators

    if nsplits <= nsplits_affecting_group_id:
        return group_ids

    # compute remaining bits
    assignments = np.copy(group_ids)
    for i in range(nsplits_affecting_group_id, nsplits):
        split = splits[i]
        vals = split.vals[group_ids]
        # x = X[:, split.dim]
        # if split.offset is not None:
        #     x = x - split.offset
        # if split.scaleby is not None:
        #     x = x * split.scaleby
        # indicators = x > vals
        indicators = split.preprocess_x(X[:, split.dim]) > vals
        assignments = (assignments * 2) + indicators

    return assignments


def _centroids_from_assignments(X, assignments, ncentroids):
    centroids = np.empty((ncentroids, X.shape[1]), dtype=X.dtype)
    for c in range(ncentroids):
        centroids[c] = X[assignments == c].mean(axis=0)
    return centroids


def centroids_from_splits(X, splits):
    ncentroids = int(1 << len(splits))
    assignments = assignments_from_splits(X, splits)
    return _centroids_from_assignments(X, assignments, ncentroids=ncentroids)


@_memory.cache
def learn_splits_in_subspaces(X, subvect_len, nsplits_per_subs,
                              return_centroids=True, algo='multisplits',
                              verbose=2):
    N, D = X.shape

    # N /= 100 # TODO rm after debug

    splits_lists = []
    nsubs = int(np.ceil(D) / subvect_len)

    # stuff for sse stats
    tot_sse = 0
    X_bar = X - np.mean(X, axis=0)
    col_sses = np.sum(X_bar * X_bar, axis=0) + 1e-14
    tot_sse_using_mean = np.sum(col_sses)
    if verbose > 1:
        print("original sum of sses within each col: ", tot_sse_using_mean)

    if return_centroids:
        ncentroids = int(2 ** nsplits_per_subs)
        # this order seems weird, but matches _learn_centroids, etc; helps with
        # eventual vectorized lookups
        centroids = np.empty((ncentroids, nsubs, subvect_len), dtype=X.dtype)

    for m in range(nsubs):
        start_col = m * subvect_len
        end_col = start_col + subvect_len
        X_subs = X[:, start_col:end_col]

        splits, sse, subs_centroids = learn_splits(
            X_subs, nsplits=nsplits_per_subs, verbose=(verbose - 1),
            return_centroids=True, algo=algo)
        centroids[:, m, :] = subs_centroids
        splits_lists.append(splits)

        tot_sse += sse
        if verbose > 1:
            # print("col sses in subspace: ", col_sses[start_col:end_col])
            # print("sum col sses in subspace: ", col_sses[start_col:end_col].sum())
            # print("buckets claim sse:", sse)
            # print("N: ", N)
            # print("(sse / N)", (sse / N))
            # print("np.var(X_subs)", np.var(X_subs))
            orig_sse_in_subs = col_sses[start_col:end_col].sum()
            # print("learning splits: mse / var(X) in subs {}/{} = {:3g}".format(
            #     m + 1, nsubs, (sse / N) / np.var(X_subs)))
            print("learning splits: sse / orig sse in subs {}/{} = {:3g}".format(
                m + 1, nsubs, sse / orig_sse_in_subs))

        # import sys; sys.exit()

        # print("exiting after one subspace")
        # import sys; sys.exit()

    if verbose > 0:
        print("-- learn_splits_in_subspaces: new / orig mse: {:.3g}".format(
            tot_sse / tot_sse_using_mean))
        # print("tot_sse_using_mean: ", tot_sse_using_mean)
    if return_centroids:
        return splits_lists, centroids
    return splits_lists


def encode_using_splits(X, subvect_len, splits_lists, split_type='single'):
    N, D = X.shape
    nsubs = int(np.ceil(D) / subvect_len)
    X_enc = np.empty((X.shape[0], nsubs), dtype=np.int, order='f')
    for m in range(nsubs):
        start_col = m * subvect_len
        end_col = start_col + subvect_len
        X_subs = X[:, start_col:end_col]
        if split_type == 'single':
            X_enc[:, m] = assignments_from_splits(X_subs, splits_lists[m])
        elif split_type == 'multi':
            X_enc[:, m] = assignments_from_multisplits(X_subs, splits_lists[m])

    return np.ascontiguousarray(X_enc)


def _plot_stuff_on_trace():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from joblib import Memory
    _memory = Memory('.', verbose=0)

    mpl.rcParams['lines.linewidth'] = .5

    @_memory.cache
    def _load_trace():
        return np.loadtxt('assets/debug/Trace/Trace_TRAIN.txt')

    # try_ndims = 128
    # try_ndims = 64
    try_ndims = 4

    # limit_n = 20
    # limit_n = 50
    # limit_n = 200
    limit_n = 500
    # X = np.loadtxt('assets/debug/Trace/Trace_TRAIN.txt')[:limit_n]
    X = _load_trace()[:limit_n]
    y = (X[:, 0] - 1).astype(np.int)
    X = X[:, 1:]

    _, axes = plt.subplots(3, 4, figsize=(13, 9), sharey=True)
    colors = ('blue', 'red', 'green', 'black')
    axes[0, 0].set_title('Trace Dataset\n(colored by class)')
    for lbl in np.unique(y):
        X_subset = X[y == lbl]
        axes[0, 0].plot(X_subset.T, color=colors[lbl])

    # visualize output with only 1 codebook (no need for updates)
    ncodebooks = 1
    splits, centroids, buckets = learn_mithral(
        X, ncodebooks, return_buckets=True, try_ndims=try_ndims, niters=1)
    centroids = centroids[0]  # only one codebook
    axes[0, 1].set_title('centroids')
    axes[0, 1].plot(centroids.T)
    X_hat = np.zeros_like(X)
    for c, splitlist in enumerate(splits):
        for s, split in enumerate(splitlist):
            assert len(splitlist) == 4
            vals = (split.vals / split.scaleby) + split.offset
            for val in vals:
                axes[0, c].scatter(split.dim, val, color=colors[s], marker='o', zorder=5)
    for b in buckets[0]:  # only one codebook, so use first list
        if b.N > 0:
            X_hat[b.point_ids] = b.col_means()
    X_res = X - X_hat
    axes[0, 2].set_title('reconstructions')
    axes[0, 2].plot(X_hat.T)
    # axes[0, 3].set_title('residuals (mean={:.2f})'.format(X_res.mean()))
    axes[0, 3].set_title('residuals (var={:.2f})'.format(X_res.var()))
    axes[0, 3].plot(X_res.T)

    # visualize output with only 2 codebooks, no updates
    ncodebooks = 2
    splits, centroids, buckets = learn_mithral(
        X, ncodebooks, return_buckets=True, try_ndims=try_ndims, niters=1)
    # centroids = centroids[0]  # only one codebook
    axes[1, 0].set_title('centroids[0]')
    axes[1, 0].plot(centroids[0].T)
    axes[1, 1].set_title('centroids[1]')
    axes[1, 1].plot(centroids[1].T)
    X_hat = np.zeros_like(X)
    # print("splits: ", splits)
    for c, splitlist in enumerate(splits):
        for s, split in enumerate(splitlist):
            assert len(splitlist) == 4
            vals = (split.vals / split.scaleby) + split.offset
            for val in vals:
                axes[1, c].scatter(split.dim, val, color=colors[s])
    for c in range(len(buckets)):  # for each codebook
        for b, buck in enumerate(buckets[c]):
            if buck.N > 0:
                X_hat[buck.point_ids] += centroids[c, b]
    X_res = X - X_hat
    axes[1, 2].set_title('reconstructions')
    axes[1, 2].plot(X_hat.T)
    # axes[1, 3].set_title('residuals (mean={:.2f})'.format(X_res.mean()))
    axes[1, 3].set_title('residuals (var={:.2f})'.format(X_res.var()))
    axes[1, 3].plot(X_res.T)

    # visualize output with only 2 codebooks, with centroid updates
    ncodebooks = 2
    splits, centroids, buckets = learn_mithral(
        X, ncodebooks, return_buckets=True, try_ndims=try_ndims, niters=1)
    axes[2, 0].set_title('centroids[0]')
    axes[2, 0].plot(centroids[0].T)
    axes[2, 1].set_title('centroids[1]')
    axes[2, 1].plot(centroids[1].T)
    X_hat = np.zeros_like(X)
    for c in range(len(buckets)):  # for each codebook
        for b, buck in enumerate(buckets[c]):
            if buck.N > 0:
                X_hat[buck.point_ids] += centroids[c, b]
    X_res = X - X_hat
    axes[2, 2].set_title('reconstructions')
    axes[2, 2].plot(X_hat.T)
    # axes[2, 3].set_title('residuals (mean={:.2f})'.format(X_res.mean()))
    axes[2, 3].set_title('residuals (var={:.2f})'.format(X_res.var()))
    axes[2, 3].plot(X_res.T)

    plt.tight_layout()
    plt.show()


def test_encoded_ops():
    N, C, K = 100, 8, 16
    X_enc = np.random.randint(K, size=(N, C))
    # print(X_enc)
    X_bin = _densify_X_enc(X_enc)
    # print(X_enc_binary)
    assert np.all(X_bin.sum(axis=1) == C)

    XtX = _XtX_encoded(X_enc)
    XtX2 = X_bin.T @ X_bin
    assert np.all(XtX == XtX2)

    M = 17
    Y = np.random.randn(N, M).astype(np.float32)

    XtY = _XtY_encoded(X_enc, Y)
    XtY2 = X_bin.T @ Y
    # print(XtY[:2])
    # print(XtY2[:2])
    assert np.all(XtY == XtY2)

    D = C * K
    W = np.random.randn(D, M).astype(np.float32)
    XW = _XW_encoded(X_enc, W)
    XW2 = X_bin @ W
    assert np.all(XW == XW2)


def main():
    test_encoded_ops()

    # print(_pq_codebook_start_end_idxs(6, 3))
    # print(_pq_codebook_start_end_idxs(8, 3))
    # print(_pq_codebook_start_end_idxs(9, 3))
    # print(_pq_codebook_start_end_idxs(10, 3))


if __name__ == '__main__':
    main()
