#!/bin/env/python

import copy
import numpy as np

from functools import reduce

# def bucket_id_to_new_bucket_ids(old_id):
#     i = 2 * old_id
#     return i, i + 1


class Bucket(object):
    __slots__ = 'N D id sumX sumX2 point_ids'.split()

    def __init__(self, D=None, N=0, sumX=None, sumX2=None, point_ids=None,
                 bucket_id=0):
        # self.reset(D=D, sumX=sumX, sumX2=sumX2)
        # assert point_ids is not None
        if point_ids is None:
            assert N == 0
            point_ids = set()

        self.N = len(point_ids)
        self.point_ids = set(point_ids)
        self.id = bucket_id

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

        # if point_ids is not None:
        #     self.point_ids = set(point_ids)
        #     if N > 0:
        #         assert N == len(point_ids)
        #     else:
        #         N = len(point_ids)
        # self.N = int(N)

        # print("created bucket with {} point ids".format(len(self.point_ids)))

    # def reset(self, D=None, sumX=None, sumX2=None):
    #     assert D is not None
    #     self.D = D
    #     self.N = 0
    #     self.point_ids = set()
    #     # assert (D is not None) or ((sumX is not None) and (sumX2 is not None))
    #     self.sumX = np.zeros(D, dtype=np.float32) if sumX is None else sumX
    #     self.sumX2 = np.zeros(D, dtype=np.float32) if sumX2 is None else sumX2

    def add_point(self, point, point_id=None):
        # TODO replace with more numerically stable updates if necessary
        self.N += 1
        self.sumX += point
        self.sumX2 += point * point
        if point_id is not None:
            self.point_ids.add(point_id)

    def remove_point(self, point, point_id=None):
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

    def split(self, X=None, dim=None, val=None):
        id0 = 2 * self.id
        id1 = id0 + 1
        if X is None or self.N < 2:  # copy of this bucket + an empty bucket
            return (self.deepcopy(bucket_id=id0),
                    Bucket(D=self.D, bucket_id=id1))
        assert dim is not None
        assert val is not None
        assert self.point_ids is not None
        my_idxs = np.array(list(self.point_ids))
        # print("my_idxs shape, dtype", my_idxs.shape, my_idxs.dtype)
        X = X[my_idxs]
        mask = X[:, dim] < val
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

    def optimal_split_val(self, X, dim, possible_vals=None):
        if self.N < 2 or self.point_ids is None:
            return 0, 0
        my_idxs = np.array(list(self.point_ids))
        return optimal_split_val(X[my_idxs], dim, possible_vals=possible_vals)

    def col_means(self):
        return self.sumX / max(1, self.N)

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


# def optimal_split_val(X, dim, possible_vals=None, return_val_idx=False):
def optimal_split_val(X, dim, possible_vals=None,
                      return_possible_vals_losses=False):
    N, D = X.shape
    sort_idxs = np.argsort(X[:, dim])
    X_sort = X[sort_idxs]
    X_sort_sq = X_sort * X_sort
    cumX_head = np.cumsum(X_sort, axis=0)
    cumX2_head = np.cumsum(X_sort_sq, axis=0)
    cumX_tail = np.cumsum(X_sort[::-1], axis=0)[::-1]
    cumX2_tail = np.cumsum(X_sort_sq[::-1], axis=0)[::-1]

    all_counts = np.arange(1, N + 1).reshape(-1, 1)
    EX_head = cumX_head / all_counts            # E[X], starting from 0
    EX_tail = cumX_tail / all_counts[::-1]      # E[X], starting from N-1
    EX2_head = cumX2_head / all_counts          # E[X^2], starting from 0
    EX2_tail = cumX2_tail / all_counts[::-1]    # E[X^2], starting from N-1
    sses_head = EX2_head - EX_head * EX_head    # sum of squares from 0
    sses_tail = EX2_tail - EX_tail * EX_tail    # sum of squares from N-1

    sses = sses_head
    sses[:-1] += sses_tail[1:]  # sse of X_sort[:i] + sse of X_sort[i:]
    sses = sses.sum(axis=1)

    if possible_vals is None or not len(possible_vals):  # can split anywhere
        best_idx = np.argmin(sses)
        next_idx = min(N - 1, best_idx + 1)
        best_val = (X_sort[best_idx, dim] + X_sort[next_idx, dim]) / 2.
    else:  # have to choose one of the values in possible_vals
        sorted_col = X_sort[:, dim]
        idxs = np.searchsorted(sorted_col, possible_vals)
        # idxs = np.unique(idxs)
        idxs = np.maximum(0, idxs - 1)  # searchsorted returns first idx larger
        sses_for_idxs = sses[idxs]
        which_idx_idx = np.argmin(sses_for_idxs)
        best_idx = idxs[which_idx_idx]
        best_val = possible_vals[which_idx_idx]

    ret = best_val, sses[best_idx]
    return ret + (sses_for_idxs,) if return_possible_vals_losses else ret


def evenly_spaced_quantiles(x, nquantiles):
    n = nquantiles + 1.
    fracs = np.arange(1, n) / n
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
    __slots__ = 'dim vals'.split()

    def __init__(self, dim, vals):
        self.dim = dim
        self.vals = vals


def learn_splits_cooler(X, nsplits, log2_max_vals_per_split=4,
                        try_nquantiles=16, return_centroids=True, verbose=3):
    N, D = X.shape
    max_vals_per_split = 1 << log2_max_vals_per_split

    # initially, one big bucket with everything
    buckets = [Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0),
               point_ids=np.arange(N))]
    total_loss = sum([bucket.loss for bucket in buckets])

    # values to try in each dim, after buckets no longer get to pick optimal
    # ones; we try values that at evenly spaced quantiles
    possible_split_vals = np.empty((D, try_nquantiles), dtype=X.dtype)
    for dim in range(D):
        possible_split_vals[dim] = evenly_spaced_quantiles(
            X[:, dim], try_nquantiles)

    if verbose > 0:
        print("learn_splits(): initial loss: ", total_loss)

    splits = []
    for s in range(nsplits):
        if verbose > 1:
            print("================================ finding split #:", s)
        for i, buck in enumerate(buckets):  # TODO rm sanity check
            assert buck.id == i

        nbuckets = len(buckets)

        # compute number of bucket groups and size of each
        ngroups = min(nbuckets, max_vals_per_split)
        nbuckets_per_group = nbuckets // ngroups
        assert nbuckets_per_group * ngroups == nbuckets  # sanity check

        try_dims = np.arange(D)  # TODO restrict to subset?
        losses = np.zeros(len(try_dims), dtype=X.dtype)
        all_split_vals = []  # vals chosen by each bucket/group for each dim

        # determine for this dim what the best split vals are for each
        # group and what the loss is when using these split vals
        for dim in try_dims:
            if verbose > 2:
                print("---------------------- dim = ", dim)
            # just let each bucket pick its optimal split val for this dim;
            # special case of below where each "group" is one bucket, and
            # instead of having to pick val from fixed set, can be anything
            if nbuckets_per_group == 1:
                split_vals = []  # each bucket contributes one split val
                for buck in buckets:
                    val, loss = buck.optimal_split_val(X, dim)
                    losses[dim] += loss
                    split_vals.append(val)
                all_split_vals.append(split_vals)
            # buckets have to pick from fixed set of possible values; each
            # group of buckets (defined by common prefix) have to agree on
            # one val, so we sum loss for each possible value across all
            # buckets in the group, and then take val with lowest sum
            else:
                split_vals = []  # each group contributes one split val
                losses = np.zeros_like(possible_split_vals[0])
                for g in range(ngroups):
                    start_idx = g * nbuckets_per_group
                    end_idx = start_idx + nbuckets_per_group
                    group_buckets = buckets[start_idx:end_idx]

                    # compute loss for each possible split value, summed
                    # across all buckets in this group; then choose best
                    losses[:] = 0
                    possible_vals = possible_split_vals[dim]
                    for b, buck in enumerate(group_buckets):
                        _, _, val_losses = optimal_split_val(
                            X, dim, possible_vals=possible_vals,
                            return_possible_vals_losses=True)
                        losses += val_losses
                    best_val_idx = np.argmin(losses)
                    best_val = possible_vals[best_val_idx]
                    best_loss = losses[best_val_idx]
                    losses[dim] += best_loss
                    split_vals.append(best_val)
                all_split_vals.append(split_vals)

        # determine best dim to split on, and pull out associated split
        # vals for all buckets
        best_tried_dim_idx = np.argmin(losses)
        best_dim = try_dims[best_tried_dim_idx]
        use_split_vals = all_split_vals[best_tried_dim_idx]
        split = MultiSplit(dim=best_dim, vals=use_split_vals)
        splits.append(split)

        # apply this split to get next round of buckets
        new_buckets = []
        for i, buck in enumerate(buckets):
            group_idx = i // nbuckets_per_group
            val = use_split_vals[group_idx]
            new_buckets += list(buck.split(X, dim=best_dim, val=val))
        buckets = new_buckets

    # maybe return centroids in addition to set of MultiSplits and loss
    loss = sum([bucket.loss for bucket in buckets])
    if return_centroids:
        centroids = np.vstack([buck.col_means() for buck in buckets])
        assert centroids.shape == (len(buckets), X.shape[1])
        return splits, loss, centroids
    return splits, loss


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


def learn_splits(X, nsplits, return_centroids=True, **kwargs):
    # indirect to particular func; will likely need to try something simpler
    # for debugging and/or as experimental control
    # return learn_splits_greedy(X, nsplits, **kwargs)
    # return learn_splits_simple(X, nsplits, **kwargs)
    # return learn_splits_conditional(X, nsplits, **kwargs)
    # return learn_splits_greedy(X, nsplits) # TODO fwd kwargs
    # splits, loss = learn_splits_greedy(X, nsplits)
    # if return_centroids:
    #     centroids = centroids_from_splits(X, splits)
    #     return splits, loss, centroids
    # return splits, loss

    return learn_splits_cooler(X, nsplits, return_centroids=return_centroids)


def assignments_from_splits(X, splits):
    nsplits = len(splits)
    indicators = np.empty((nsplits, len(X)), dtype=np.int)
    for i, split in enumerate(splits):
        indicators[i] = X[:, split.dim] > split.val

    # compute assignments by treating indicators in a row as a binary num
    # scales = (2 ** np.arange(nsplits)).astype(np.int)
    scales = (1 << np.arange(nsplits)).astype(np.int)
    return (indicators.T * scales).sum(axis=1).astype(np.int)


def _centroids_from_assignments(X, assignments, ncentroids):
    centroids = np.empty((ncentroids, X.shape[1]), dtype=X.dtype)
    for c in range(ncentroids):
        centroids[c] = X[assignments == c].mean(axis=0)
    return centroids


def centroids_from_splits(X, splits):
    ncentroids = int(1 << len(splits))
    assignments = assignments_from_splits(X, splits)
    return _centroids_from_assignments(X, assignments, ncentroids=ncentroids)


# def centroids_from_multisplits(X, splits):
#     pass

def learn_splits_in_subspaces(X, subvect_len, nsplits_per_subs,
                              return_centroids=True, verbose=2):
    N, D = X.shape
    splits_lists = []
    nsubs = int(np.ceil(D) / subvect_len)

    # stuff for sse stats
    tot_sse = 0
    X_bar = X - np.mean(X, axis=0)
    col_sses = np.sum(X_bar * X_bar, axis=0) + 1e-14
    tot_sse_using_mean = np.sum(col_sses)

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
            return_centroids=True)
        centroids[:, m, :] = subs_centroids
        splits_lists.append(splits)

        tot_sse += sse
        if verbose > 1:
            print("learning splits: mse / var(X) in subs {}/{} = {:3g}".format(
                m + 1, nsubs, (sse / N) / np.var(X_subs)))

    print("-- total mse / var(X): {:.3g}".format(tot_sse / tot_sse_using_mean))
    if return_centroids:
        return splits_lists, centroids
    return splits_lists


def encode_using_splits(X, subvect_len, splits_lists):
    N, D = X.shape
    nsubs = int(np.ceil(D) / subvect_len)
    X_enc = np.empty((X.shape[0], nsubs), dtype=np.int, order='f')
    for m in range(nsubs):
        start_col = m * subvect_len
        end_col = start_col + subvect_len
        X_subs = X[:, start_col:end_col]
        X_enc[:, m] = assignments_from_splits(X_subs, splits_lists[m])

    return np.ascontiguousarray(X_enc)


def main():
    # np.random.seed(123)
    np.random.seed(1234)
    X = np.random.randint(5, size=(5, 3)).astype(np.float32)
    print("X:\n", X)

    splits, loss = learn_splits(X, 2)

    # print('loss: ', np.var(X, axis=0))
    print('final loss: ', loss)
    # print(X)


if __name__ == '__main__':
    main()
