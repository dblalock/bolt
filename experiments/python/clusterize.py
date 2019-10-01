#!/bin/env/python

import copy
import numpy as np


# def bucket_id_to_new_bucket_ids(old_id):
#     i = 2 * old_id
#     return i, i + 1


class Bucket(object):
    __slots__ = 'N sumX sumX2 point_ids'.split()

    def __init__(self, D=None, sumX=None, sumX2=None, N=0, point_ids=None):
        self.reset(D=D, sumX=sumX, sumX2=sumX2)
        if point_ids is not None:
            self.point_ids = set(point_ids)
            if N > 0:
                assert N == len(point_ids)
            else:
                N = len(point_ids)
        self.N = int(N)

    def reset(self, D=None, sumX=None, sumX2=None):
        self.N = 0
        self.point_ids = set()
        assert (D is not None) or ((sumX is not None) and (sumX2 is not None))
        self.sumX = np.zeros(D, dtype=np.float32) if sumX is None else sumX
        self.sumX2 = np.zeros(D, dtype=np.float32) if sumX2 is None else sumX2

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

    @property
    def loss(self):
        if self.N < 1:
            return 0

        # # less stable version with one less divide and mul
        # return max(0, np.sum(self.sumX2 - (self.sumX * (self.sumX / self.N))))

        # more stable version, that also clamps variance at 0
        expected_X = self.sumX / self.N
        expected_X2 = self.sumX2 / self.N
        return max(0, np.sum(expected_X2 - (expected_X * expected_X)) * self.N)


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
    new_buckets = []
    D = len(buckets[0].sumX)
    for bucket in buckets:
        buck0 = copy.deepcopy(bucket)
        buck1 = Bucket(D=D)
        new_buckets.append((buck0, buck1))
    return new_buckets


def learn_splits_greedy(X, nsplits, verbose=2):
    N, D = X.shape
    assert nsplits <= D

    # precompute sorted lists of values within each dimension,
    # along with which row they originally were so look can look
    # up the whole vector (and bucket) associated with each value
    dim2sorted = []
    for dim in range(D):
        sorted_with_idx = _sort_and_append_orig_idx(X[:, dim])
        dim2sorted.append(sorted_with_idx)

    splits = []
    buckets = [Bucket(N=X.shape[0], sumX=X.sum(axis=0),
                      sumX2=(X * X).sum(axis=0), point_ids=np.arange(N))]

    all_point_infos = [PointInfo(data=row, bucket_id=0) for row in X]

    # Z = X - X.mean(axis=0)
    # total_loss = np.sum(Z * Z)
    # print("initial SSE: ", total_loss)

    total_loss = sum([bucket.loss for bucket in buckets])
    if verbose > 0:
        print("learn_splits(): initial loss: ", total_loss)

    unused_dims = set(np.arange(X.shape[1]))

    for s in range(nsplits):
        if verbose > 2:
            print("================================ finding split #:", s)
        best_split = Split(dim=-1, val=-np.inf, loss_change=0)
        for d in unused_dims:
            vals_and_point_ids = dim2sorted[d]
            new_buckets = _split_existing_buckets(buckets)
            new_total_loss = total_loss
            if verbose > 2:
                print("---------------------- dim = ", d)
            for val, point_id in vals_and_point_ids:
                info = all_point_infos[point_id]
                point, bucket_id = info.data, info.bucket_id

                bucket0 = new_buckets[bucket_id][0]
                bucket1 = new_buckets[bucket_id][1]
                old_loss = bucket0.loss + bucket1.loss
                bucket0.remove_point(point, point_id=point_id)
                bucket1.add_point(point, point_id=point_id)

                new_loss = bucket0.loss + bucket1.loss
                new_total_loss += new_loss - old_loss
                loss_change = new_total_loss - total_loss

                assert loss_change <= 1e-10  # should be nonincreasing

                if verbose > 2:
                    print("-------- split point_id, val = ", point_id, val)
                    print("bucket0 point ids, loss after update: ",
                          bucket0.point_ids, bucket0.loss)
                    print("bucket1 point ids, loss after update: ",
                          bucket1.point_ids, bucket1.loss)
                    print("loss change = {:.3f};\tnew_loss = {:.3f} ".format(
                          loss_change, new_total_loss))

                if loss_change < best_split.loss_change:
                    best_split.dim = d
                    best_split.val = val
                    best_split.loss_change = loss_change

        if verbose > 2:
            print("---------------------- split on dim={}, val={:.3f} ".format(
                best_split.dim, best_split.val))

        # we've identified the best split; now apply it
        new_buckets = [Bucket(D=D) for _ in 1 + np.arange(2 * len(buckets))]
        for i, info in enumerate(all_point_infos):
            # determine which bucket this point should be in
            data, bucket_id = info.data, info.bucket_id
            new_bucket_id = 2 * bucket_id
            if (data[best_split.dim] <= best_split.val):
                new_bucket_id += 1
            # update info for this point + stats for that bucket
            info.bucket_id = new_bucket_id
            new_buckets[new_bucket_id].add_point(data, point_id=i)

            if verbose > 2:
                print("applying split to point {}: {}".format(i, data))
                print("goes in bucket ", new_bucket_id)

        buckets = new_buckets
        total_loss = sum([bucket.loss for bucket in buckets])
        unused_dims.remove(best_split.dim)
        splits.append(best_split)

        if verbose > 1:
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

    return splits, total_loss


def learn_splits(X, nsplits, **kwargs):
    # indirect to particular func; will likely need to try something simpler
    # for debugging and/or as experimental control
    return learn_splits_greedy(X, nsplits, **kwargs)


def assignments_from_splits(X, splits):
    nsplits = len(splits)
    indicators = np.empty((nsplits, len(X)), dtype=np.int)
    for i, split in enumerate(splits):
        indicators[i] = X[:, split.dim] > split.val

    # compute assignments by treating indicators in a row as a binary num
    scales = int(2 ** np.arange(nsplits))
    return (indicators.T * scales).sum(axis=1)  # N x nsplits, sum rows


def _centroids_from_assignments(X, assignments, ncentroids):
    centroids = np.empty((ncentroids, X.shape[1]), dtype=X.dtype)
    for c in range(ncentroids):
        centroids[c] = X[assignments == c].mean(axis=0)
    return centroids


def centroids_from_splits(X, splits):
    ncentroids = int(2 ** len(splits))
    assignments = assignments_from_splits(X, splits)
    return _centroids_from_assignments(X, assignments, ncentroids=ncentroids)


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
        splits, sse = learn_splits(
            X_subs, nsplits=nsplits_per_subs, verbose=(verbose - 1))
        splits_lists.append(splits)
        if return_centroids:
            centroids[:, m, :] = centroids_from_splits(X_subs, splits)

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
