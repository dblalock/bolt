#!/bin/env/python

import copy
import numpy as np


# def bucket_id_to_new_bucket_ids(old_id):
#     i = 2 * old_id
#     return i, i + 1


class BucketStats(object):
    __slots__ = 'N sumX sumX2'.split()

    def __init__(self, sumX=None, sumX2=None, D=None):
        self.reset(sumX=sumX, sumX2=sumX2, D=D)

    def reset(self, sumX=None, sumX2=None, D=None):
        self.N = 0
        assert (D is not None) or (None not in (sumX, sumX2))
        self.sumX = np.zeros(D, dtype=np.float32) if sumX is None else sumX
        self.sumX2 = np.zeros(D, dtype=np.float32) if sumX2 is None else sumX2

    def add_point(self, point):
        # TODO replace with more numerically stable updates if necessary
        self.N += 1
        self.sumX += point
        self.sumX2 += point * point

    def remove_point(self, point):
        self.N -= 1
        self.sumX -= point
        self.sumX2 -= point * point

    @property
    def loss(self):
        # less stable version with one less divide
        # return self.sumX2 - (self.sumX * (self.sumX / self.N))

        # more stable version, that also clamps variance at 0
        expected_X = self.sumX / self.N
        expected_X2 = self.sumX2 / self.N
        return max(0, expected_X2 - (expected_X * expected_X))


class PointInfo(object):
    __slots__ = 'data bucket_id'.split()

    def __init__(self, data, bucket_id):
        self.data = data
        self.bucket_id = bucket_id


class Split(object):
    __slots__ = 'dim val dLoss'.split()

    def __init__(self, dim, val, dLoss=None):
        self.dim = dim
        self.val = val
        self.dLoss = dLoss


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
        buck1 = BucketStats(D=D)
        new_buckets.append((buck0, buck1))
    return new_buckets


def learn_splits(X, nsplits):
    N, D = X.shape

    # precompute sorted lists of values within each dimension,
    # along with which row they originally were so look can look
    # up the whole vector (and bucket) associated with each value
    dim2sorted = []
    for dim in range(D):
        sorted_with_idx = _sort_and_append_orig_idx(X[:, dim])
        dim2sorted.append(sorted_with_idx)

    splits = []
    buckets = [BucketStats(N=X.shape[0], D=X.shape[1])]
    all_point_infos = [PointInfo(data=row, bucket_id=0) for row in X]
    Z = X - X.mean(axis=0)
    total_loss = np.sum(Z * Z)
    unused_dims = set(np.arange(X.shape[1]))

    for _ in range(nsplits):
        best_split = Split(dim=-1, val=-np.inf, dLoss=0)
        for d in unused_dims:
            vals_and_point_ids = dim2sorted[d]
            new_buckets = _split_existing_buckets(buckets)
            new_total_loss = total_loss
            for val, point_id in vals_and_point_ids:
                info = all_point_infos[point_id]
                point, bucket_id = info.data, info.bucket_id
                # new_bucket_ids = bucket_id_to_new_bucket_ids(bucket_id)
                # bucket0 = new_buckets[new_bucket_ids[0]]
                # bucket1 = new_buckets[new_bucket_ids[1]]
                bucket0 = new_buckets[bucket_id][0]
                bucket1 = new_buckets[bucket_id][1]
                old_loss = bucket0.loss + bucket1.loss
                bucket0.remove_point(point)
                bucket1.add_point(point)
                new_loss = bucket0.loss + bucket1.loss

                # TODO loss change has to compute new total_loss; if you
                # just look at effect of adding this one point, loss change
                # is just marginal change of shifting split point by one, not
                # of applying the split at all; these are almost unrelated
                # quantities
                # loss_change = new_loss - old_loss

                new_total_loss += new_loss - old_loss
                loss_change = new_total_loss - total_loss

                if loss_change < best_split.loss_change:
                    best_split.dim = d
                    best_split.val = val
                    best_split.loss_change = loss_change

        # we've identified the best split; now apply it
        new_buckets = [BucketStats() for _ in 1 + np.arange(2 * len(buckets))]
        for info in all_point_infos:
            point, bucket_id = info.data, info.bucket_id
            # new_bucket_ids = bucket_id_to_new_bucket_ids(bucket_id)
            # which_new_bucket = 0 if (point[best_split.dim] <= val) else 1
            # reassign point to its new (more granular) bucket
            # new_bucket_id = new_bucket_ids[which_new_bucket]

            # determine which bucket this point should be in
            new_bucket_id = 2 * bucket_id
            if (point[best_split.dim] <= val):
                new_bucket_id += 1
            info.bucket_id = new_bucket_id
            # update stats for that bucket
            new_buckets[new_bucket_id].add_point(point)

        buckets = new_buckets
        total_loss = sum([bucket.loss for bucket in buckets])
        unused_dims -= best_split.dim
        splits.append(best_split)

    return splits


def main():
    X = np.random.randint(5, size=(100, 3)).astype(np.float32)
    print(X)


if __name__ == '__main__':
    main()
