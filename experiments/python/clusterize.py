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
    buckets = [Bucket(N=X.shape[0], sumX=X.sum(axis=0),
                      sumX2=(X * X).sum(axis=0), point_ids=np.arange(N))]

    all_point_infos = [PointInfo(data=row, bucket_id=0) for row in X]

    # Z = X - X.mean(axis=0)
    # total_loss = np.sum(Z * Z)
    # print("initial SSE: ", total_loss)

    total_loss = sum([bucket.loss for bucket in buckets])
    print("initial loss: ", total_loss)

    unused_dims = set(np.arange(X.shape[1]))

    for s in range(nsplits):
        print("================================ finding split #:", s)
        best_split = Split(dim=-1, val=-np.inf, loss_change=0)
        for d in unused_dims:
            vals_and_point_ids = dim2sorted[d]
            new_buckets = _split_existing_buckets(buckets)
            new_total_loss = total_loss
            print("------------------------ dim = ", d)
            for val, point_id in vals_and_point_ids:
                info = all_point_infos[point_id]
                point, bucket_id = info.data, info.bucket_id

                print("-------- split point_id, val = ", point_id, val)

                # new_bucket_ids = bucket_id_to_new_bucket_ids(bucket_id)
                # bucket0 = new_buckets[new_bucket_ids[0]]
                # bucket1 = new_buckets[new_bucket_ids[1]]
                bucket0 = new_buckets[bucket_id][0]
                bucket1 = new_buckets[bucket_id][1]
                # print("bucket0 N, loss before update: ", bucket0.N, bucket0.loss)
                old_loss = bucket0.loss + bucket1.loss
                bucket0.remove_point(point, point_id=point_id)
                bucket1.add_point(point, point_id=point_id)
                print("bucket0 point ids, loss after update: ", bucket0.point_ids, bucket0.loss)
                print("bucket1 point ids, loss after update: ", bucket1.point_ids, bucket1.loss)
                # bucket0.remove_point(point)
                # bucket1.add_point(point)
                new_loss = bucket0.loss + bucket1.loss

                new_total_loss += new_loss - old_loss
                loss_change = new_total_loss - total_loss

                print("loss change = {:.3f};\tnew_loss = {:.3f} ".format(
                      loss_change, new_total_loss))

                if loss_change < best_split.loss_change:
                    best_split.dim = d
                    best_split.val = val
                    best_split.loss_change = loss_change

        print("------------------------ split on dim={}, val={:.3f} ".format(
            best_split.dim, best_split.val))

        # we've identified the best split; now apply it
        new_buckets = [Bucket(D=D) for _ in 1 + np.arange(2 * len(buckets))]
        for i, info in enumerate(all_point_infos):
            data, bucket_id = info.data, info.bucket_id
            # new_bucket_ids = bucket_id_to_new_bucket_ids(bucket_id)
            # which_new_bucket = 0 if (point[best_split.dim] <= val) else 1
            # reassign point to its new (more granular) bucket
            # new_bucket_id = new_bucket_ids[which_new_bucket]

            print("applying split to point {}: {}".format(i, data))

            # determine which bucket this point should be in
            new_bucket_id = 2 * bucket_id
            if (data[best_split.dim] <= best_split.val):
                new_bucket_id += 1
            print("goes in bucket ", new_bucket_id)
            info.bucket_id = new_bucket_id
            # update stats for that bucket
            new_buckets[new_bucket_id].add_point(data, point_id=i)

        buckets = new_buckets

        print('bucket losses: ')
        print([bucket.loss for bucket in buckets])
        print('bucket N, sumX, sumX2')
        print([bucket.N for bucket in buckets])
        print([list(bucket.sumX) for bucket in buckets])
        print([list(bucket.sumX2) for bucket in buckets])

        total_loss = sum([bucket.loss for bucket in buckets])
        unused_dims.remove(best_split.dim)
        splits.append(best_split)

        print('new loss: {:.3f} from split at dim {}, value {:.3f}'.format(
              total_loss, best_split.dim, best_split.val))

    return splits, total_loss


def main():
    # np.random.seed(123)
    np.random.seed(1234)
    X = np.random.randint(5, size=(3, 2)).astype(np.float32)
    print("X:\n", X)

    splits, loss = learn_splits(X, 1)


    # print('loss: ', np.var(X, axis=0))
    print('final loss: ', loss)
    # print(X)


if __name__ == '__main__':
    main()
