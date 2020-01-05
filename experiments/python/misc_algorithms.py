#!#!/bin/env/python

from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .utils import kmeans

from joblib import Memory
_memory = Memory('.', verbose=0)


def _to_np(A):
    return A.cpu().detach().numpy()


def _class_balanced_sampling(X, labels, k):
    np.random.seed(123)
    N, D = X.shape

    # intialize centroids by sampling from each class in proportion to its
    # relative frequency
    uniq_lbls, counts = np.unique(labels, return_counts=True)
    sort_idxs = np.argsort(counts)
    uniq_lbls = uniq_lbls[sort_idxs]
    counts = counts[sort_idxs]
    remaining_counts = np.cumsum(counts[::-1])[::-1]
    nremaining_samples = k

    # C = np.empty((k, D), dtype=np.float32)
    C = []
    C_labels = []
    # affinities = np.zeros((k, nclasses), dtype=np.float32)

    for i, lbl in enumerate(uniq_lbls):
        count = counts[i]
        target_frac = count / remaining_counts[i]
        target_nsamples = int(nremaining_samples * target_frac + .999)
        target_nsamples = max(1, target_nsamples)
        target_nsamples = min(target_nsamples, count)
        nremaining_samples -= target_nsamples

        lbl_idxs = np.where(labels == lbl)[0]
        # print("lbl, count, num lbl idxs: ", lbl, count, len(lbl_idxs))
        assert len(lbl_idxs) == count
        use_idxs = np.random.choice(count, size=target_nsamples, replace=False)
        keep_idxs = lbl_idxs[use_idxs]
        C.append(X[keep_idxs])
        C_labels.append(np.full(target_nsamples, lbl, dtype=np.int32))
    # if len(C).shape[0] < k:
    C = np.vstack(C).astype(np.float32)

    # print("k, C shape", k, C.shape)
    assert C.shape == (k, D)
    C_labels = np.hstack(C_labels)
    assert C_labels.shape == (k,)

    return C, C_labels


def neighbor_compression(X, labels, k, niters=1000, rel_tol=.0001, verbose=1):
    N, D = X.shape

    # one-hot encode labels
    nclasses = len(np.unique(labels))
    # Y = np.zeros((N, nclasses), dtype=np.float32)
    # for i in range(N):
    #     Y[i, labels[i]] = 1

    # intialize centroids
    # C, _ = kmeans(X, k)
    C, C_labels = _class_balanced_sampling(X, labels, k)

    # convert to torch tensors for optimization
    # Y = torch.from_numpy(Y)
    C = torch.tensor(C.T, requires_grad=True)  # not from_numpy to allow grad
    X = torch.from_numpy(X)
    # having trained class affinities doesn't really seem to help
    # Z = torch.randn(k, nclasses, requires_grad=True)

    # print("uniq labels: ", np.unique(labels))
    # print("uniq C_labels: ", np.unique(C_labels))

    # one-hot encode labels
    affinities = torch.zeros((k, nclasses),
                             dtype=torch.float32, requires_grad=True)
    for kk in range(k):
        affinities[kk, C_labels[kk]] = 1
    Z = affinities.clone().detach().requires_grad_(True)

    labels = torch.from_numpy(labels)

    loss_fn = torch.nn.CrossEntropyLoss()
    # opt = optim.SGD([C], lr=.1, momentum=.9)
    # opt = optim.SGD([C, affinities], lr=.1, momentum=.9)
    opt = optim.SGD([C, Z], lr=.1, momentum=.9)

    # X_norms_sq = (X * X).sum(dim=1).view(-1, 1)
    prev_loss = np.inf
    for t in range(niters):
        temperature = np.log2(t + 2)  # +2 so that it starts at 1 at t=0

        # # compute distances to all centroids
        # # prods = torch.mm(X, C)
        # prods = X @ C
        # # norms_sq = torch.sqrt(torch.sum(C * C))
        # # dists_sq = prods - norms_sq
        # # C_norms_sq = torch.sqrt(torch.sum(C * C, dim=0))
        # # C_norms_sq = torch.sum(C * C, dim=0)
        # # dists_sq = -2 * prods
        # # dists_sq += X_norms_sq
        # # dists_sq += C_norms_sq
        # # neg_dists_sq = -dists_sq
        # neg_dists_sq = prods

        # # # update soft labels for each centroid
        # # similarities = F.softmax(neg_dists_sq, dim=0)  # N x C; sim to each sample
        # # class_affinities = similarities.transpose(0, 1) @ Y  # C x nclasses
        # # class_affinities = F.softmax(class_affinities * temperature, dim=1)

        # # update class assignments for inputs
        # # centroid_similarities = F.softmax(neg_dists_sq * temperature, dim=1)  # N x C
        # centroid_similarities = F.softmax(neg_dists_sq, dim=1)  # N x C
        # # centroid_similarities = torch.exp(neg_dists_sq / np.sqrt(D))
        # # logits = centroid_similarities @ class_affinities
        # logits = centroid_similarities @ Z

        # way simpler version
        similarities = F.softmax(X @ C, dim=1)  # N x C
        # logits = similarities @ Z
        affinities = F.softmax(Z * temperature, dim=1)
        # affinities = F.softmax(affinities * temperature, dim=1)
        logits = similarities @ affinities

        # update params and print how we're doing
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()

        loss_pyfloat = loss.item()
        change = prev_loss - loss_pyfloat
        thresh = rel_tol * min(loss_pyfloat, prev_loss)
        if np.abs(change) < thresh:
            if verbose > 0:
                _, labels_hat = torch.max(logits, dim=1)
                acc = torch.mean((labels == labels_hat).type(torch.float))
                print("converged after {} iters with acc {:.3f}, loss: {:.4f}"
                      "".format(t + 1, acc.item(), loss_pyfloat))
            break  # converged
        prev_loss = loss_pyfloat

        if (verbose > 1) and ((t + 1) % 10 == 0):
            _, labels_hat = torch.max(logits, dim=1)
            acc = torch.mean((labels == labels_hat).type(torch.float)).item()
            print("acc: ", acc)
            print("{:.3f}".format(loss.item()))  # convert to python float

    # return _to_np(C).T, _to_np(class_affinities)

    centroid_labels = np.argmax(_to_np(Z), axis=1)
    return _to_np(C).T, centroid_labels


# or at least, ProtoNN without the L0 constraints; also with simultaneous
# updates to all param tensors instead of alternating
# def protonn(X, labels, k, niters=10000, verbose=1, gamma=1):
def protonn(X, labels, k, d=-1, niters=1000, verbose=1, gamma=-1):
    N, D = X.shape
    if gamma < 1:
        gamma = 1. / np.sqrt(D)  # makes it struggle less / not make NaNs
        # gamma = 1. / D
    if d < 1:
        d = D

    labels = torch.from_numpy(labels)

    # # one-hot encode labels
    nclasses = len(np.unique(labels))
    # Y = np.zeros((N, nclasses), dtype=np.float32)
    # for i in range(N):
    #     Y[i, labels[i]] = 1

    # intialize centroids
    C, _ = kmeans(X, k)
    W = np.random.randn(D, d).astype(np.float32)
    # C = C @ W
    # W = np.eye(D).astype(np.float32)[:, :d]  # better than randn init

    # convert to torch tensors for optimization
    # Y = torch.from_numpy(Y)
    C = torch.tensor(C.T, requires_grad=True)  # not from_numpy to allow grad
    X = torch.from_numpy(X)
    W = torch.tensor(W, requires_grad=True)  # not from_numpy to allow grad
    # gamma = torch.tensor(np.array(gamma, dtype=np.float32), requires_grad=True)
    # labels = torch.from_numpy(labels)
    # print("W", W[:10])
    # return None, None, None

    Z = torch.randn(k, nclasses, requires_grad=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    # opt = optim.SGD([C, Z], lr=.1, momentum=.9)
    opt = optim.SGD([C, W, Z], lr=.1, momentum=.9)
    # opt = optim.SGD([C, W, Z, gamma], lr=.1, momentum=.9)

    nbatches = 1
    batch_sz = int(np.ceil(N / nbatches))
    # batch_sz = 1024
    # nbatches = int(np.ceil(N / batch_sz))

    # for t in range(1):
    for t in range(niters):
        perm = np.random.permutation(N)
        for b in range(nbatches):
            start_idx = b * batch_sz
            end_idx = min(start_idx + batch_sz, N)
            perm_idxs = perm[start_idx:end_idx]

            X_batch = X[perm_idxs]
            labels_batch = labels[perm_idxs]

            # temperature = np.log2(t + 2)  # +2 so that it starts at 1 at t=0

            # compute distances to all centroids
            # embeddings = X @ W
            # embeddings = X_batch @ W
            embeddings = X_batch
            embed_norms_sq = (embeddings * embeddings).sum(dim=1, keepdim=True)

            # prods = torch.mm(embeddings, C)
            prods = embeddings @ C
            C_norms_sq = torch.sum(C * C, dim=0)
            dists_sq = -2 * prods
            dists_sq += embed_norms_sq
            dists_sq += C_norms_sq
            neg_dists_sq = -dists_sq
            # print("gamma: ", gamma)
            # use_gamma = torch.clamp(gamma, max=1.)
            # use_gamma = torch.clamp(gamma, 0, 1)
            # use_gamma = F.sigmoid(gamma)
            # gamma = torch.min((1, gamma))
            # gamma = torch.max((0, gamma))

            assert np.min(_to_np(dists_sq)) >= 0
            assert np.max(_to_np(neg_dists_sq)) <= 0
            similarities = torch.exp(gamma * neg_dists_sq)  # N x C
            # similarities = torch.exp(use_gamma * neg_dists_sq)  # N x C

            logits = similarities @ Z
            # print("logits shape: ", logits.shape)
            # print("logits shape: ", logits.shape)
            # logits_np = _to_np(logits)
            # print("dists_sq shape", dists_sq.shape)
            # print("dists_sq", dists_sq[:10])
            # print("C_norms_sq", C_norms_sq)
            # print("embed_norms_sq", embed_norms_sq[:10])
            # print("similarities", similarities[:10])
            # print("logits", logits[:10])

            # update soft labels for each centroid
            # similarities = F.softmax(neg_dists_sq, dim=0)  # N x C; sim to each sample
            # class_affinities = similarities.transpose(0, 1) @ Y  # C x nclasses
            # class_affinities = F.softmax(class_affinities * temperature, dim=1)

            # # update class assignments for inputs
            # centroid_similarities = F.softmax(neg_dists_sq * temperature, dim=1)  # N x C
            # logits = centroid_similarities @ affinities

            # update params and print how we're doing
            # loss = loss_fn(logits, labels)
            loss = loss_fn(logits, labels_batch)
            # loss += .01 * (gamma * gamma).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            # if (verbose > 0) and (t % 10 == 0):
            # if (verbose > 0) and ((t + 1) % 10 == 0):
            if (verbose > 0) and ((t + 1) % 10 == 0) and b == 0:
                _, labels_hat = torch.max(logits, dim=1)
                acc = torch.mean((labels[perm_idxs] == labels_hat).type(torch.float))
                print("acc: ", acc)
                print("{:.3f}".format(loss.item()))  # convert to python float
                # print("gamma: ", gamma.item())

    return _to_np(C).T, _to_np(W), _to_np(Z)


@_memory.cache
def stochastic_neighbor_compression(X, labels, k, niters=1000,
                                    gamma=-1, rel_tol=.0001, verbose=1):
    N, D = X.shape
    nclasses = len(np.unique(labels))
    if gamma < 1:
        gamma = 1
        # gamma = 1. / np.sqrt(D)  # makes it struggle less / not make NaNs
        # gamma = 1. / D

    # labels = torch.from_numpy(labels)

    # C = np.random.randn(k, D).astype(np.float32)
    C, C_labels = _class_balanced_sampling(X, labels, k)

    # one-hot encode labels
    affinities = torch.zeros((k, nclasses), dtype=torch.float32)
    for kk in range(k):
        affinities[kk, C_labels[kk]] = 1

    # so that there's actual gradient flow
    affinities += torch.randn(k, nclasses) * .1

    # W = np.random.randn(D, D).astype(np.float32)
    # C = C @ W
    # W = np.eye(D).astype(np.float32)  # better than randn init

    # convert to torch tensors for optimization
    # Y = torch.from_numpy(Y)
    C = torch.tensor(C.T, requires_grad=True)  # not from_numpy to allow grad
    X = torch.from_numpy(X)
    labels = torch.from_numpy(labels)
    gamma = torch.tensor(np.array(gamma, dtype=np.float32))
    # affinities = torch.from_numpy(affinities)
    # print("labels shape: ", labels.shape)
    # print("uniq labels: ", uniq_lbls)
    # print("uniq label counts: ", counts)
    # labels = labels.reshape(-1, 1)
    # print("labels shape: ", labels.shape)
    # W = torch.tensor(W, requires_grad=True)  # not from_numpy to allow grad
    # print("W", W[:10])
    # return None, None, None

    # Z = torch.randn(k, nclasses, requires_grad=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = optim.SGD([C], lr=.1, momentum=.9)
    # opt = optim.SGD([C, Z], lr=.1, momentum=.9)
    # opt = optim.SGD([C, gamma], lr=.1, momentum=.9)

    nbatches = 1
    batch_sz = int(np.ceil(N / nbatches))
    # batch_sz = 1024
    # nbatches = int(np.ceil(N / batch_sz))

    # for t in range(50):
    prev_loss = np.inf
    converged = False
    t = 0
    while t < niters and not converged:
        perm = np.random.permutation(N)
        for b in range(nbatches):
            if nbatches > 1:
                start_idx = b * batch_sz
                end_idx = min(start_idx + batch_sz, N)
                perm_idxs = perm[start_idx:end_idx]

                X_batch = X[perm_idxs]
                labels_batch = labels[perm_idxs]
            else:
                X_batch = X
                labels_batch = labels

            # temperature = np.log2(t + 2)  # +2 so that it starts at 1 at t=0

            # compute distances to all centroids
            # embeddings = X @ W
            # embeddings = X_batch @ W
            embeddings = X_batch
            embed_norms_sq = (embeddings * embeddings).sum(dim=1, keepdim=True)

            # prods = torch.mm(embeddings, C)
            prods = embeddings @ C
            C_norms_sq = torch.sum(C * C, dim=0)
            dists_sq = -2 * prods
            dists_sq += embed_norms_sq
            dists_sq += C_norms_sq
            neg_dists_sq = -dists_sq
            # print("min dist sq: ", torch.min(dists_sq).item())
            minval_dist_sq = torch.min(dists_sq).item()
            if minval_dist_sq < -.01:
                print("min dist sq: ", minval_dist_sq)
                print("min C_norms_sq", torch.min(C_norms_sq).item())
                print("min X_norms_sq", torch.min(embed_norms_sq).item())
                print("dists_sq: ", dists_sq[:10])
            assert minval_dist_sq >= -.01
            # assert np.min(_to_np(dists_sq)) >= -1e-3
            # assert np.max(_to_np(neg_dists_sq)) <= 1e-3
            similarities = torch.exp(gamma * neg_dists_sq)  # N x C

            logits = similarities @ affinities
            # logits = similarities @ Z

            # print("logits shape: ", logits.shape)
            # print("logits shape: ", logits.shape)
            # print("dists_sq shape", dists_sq.shape)
            # print("dists_sq", dists_sq[:10])
            # print("C_norms_sq", C_norms_sq)
            # print("embed_norms_sq", embed_norms_sq[:10])
            # print("similarities", similarities[:10])
            # print("logits", logits[:10])

            # update params and print how we're doing
            loss = loss_fn(logits, labels_batch)
            # loss += gamma * gamma
            loss.backward()
            opt.step()
            opt.zero_grad()

            loss_pyfloat = loss.item()
            change = prev_loss - loss_pyfloat
            thresh = rel_tol * min(loss_pyfloat, prev_loss)
            if np.abs(change) < thresh:
                if verbose > 0:
                    _, labels_hat = torch.max(logits, dim=1)
                    labels_true = labels[perm_idxs] if nbatches > 1 else labels
                    acc = torch.mean(
                        (labels_true == labels_hat).type(torch.float))
                    print("converged after {} iters with acc {:.3f}, loss: {:.4f}" # noqa
                          "".format(t + 1, acc.item(), loss_pyfloat))
                converged = True  # converged
                break
            prev_loss = loss_pyfloat

            # if (verbose > 0) and ((t + 1) % 10 == 0):
            # if (verbose > 0) and ((t + 1) % 10 == 0) and b == 0:
            if (verbose > 1) and (t % 10 == 0) and b == 0:
                _, labels_hat = torch.max(logits, dim=1)
                labels_true = labels[perm_idxs] if nbatches > 1 else labels
                acc = torch.mean(
                    (labels_true == labels_hat).type(torch.float))
                print("acc: {:.3f}".format(acc.item()))
                print("{:.3f}".format(loss.item()))  # convert to python float
                # print("gamma: ", gamma.item())
        t += 1

    return _to_np(C).T, C_labels


def linear_regression_log_loss(
        X, Y, lamda=1, max_niters=1000, rel_tol=.0001, verbose=2):
    N, D = X.shape
    N, M = Y.shape
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    # initialize W to OLS solution
    XtX = X.T @ X
    XtX += np.eye(D) * np.std(X)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY).astype(np.float32)
    # W += np.random.randn(*W.shape)

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    W = torch.tensor(W, requires_grad=True)
    # W = torch.randn(D, M, requires_grad=True)
    # W += torch.randn(D, M, requires_grad=False)
    opt = optim.SGD([W], lr=.1, momentum=.9)

    # now optimize using pytorch
    prev_loss = np.inf
    for t in range(max_niters):
        Y_hat = X @ W
        diffs = Y - Y_hat
        # errs = torch.floor(torch.abs(diffs))
        # loss = torch.abs(diffs)  # TODO rm
        # loss = diffs * diffs

        loss = torch.log2(1 + torch.abs(diffs))
        # loss = torch.log2(1e-10 + torch.abs(diffs))
        # loss *= (loss > 0).type(torch.float32)
        loss = torch.mean(loss)
        # loss = torch.max(loss, 0)

        loss.backward()
        opt.step()
        opt.zero_grad()

        loss_pyfloat = loss.item()
        change = prev_loss - loss_pyfloat
        thresh = rel_tol * min(loss_pyfloat, prev_loss)
        if np.abs(change) < thresh:
            if verbose > 0:
                print("converged after {} iters with loss: {:.4f}".format(
                    t + 1, loss_pyfloat))
            break  # converged
        prev_loss = loss_pyfloat

        if (verbose > 1) and ((t + 1) % 10 == 0):
            print("loss: {:.4f}".format(loss_pyfloat))

    return _to_np(W)


def main():
    # N, D = 10000, 20
    N, D = 1000, 20
    # niters = 1000
    niters = 10000
    X = np.random.randn(N, D).astype(np.float32)

    # ------------------------ linear regression with weird loss
    # M = 20
    # Y = np.random.randn(N, M).astype(np.float32)
    # linear_regression_log_loss(X, Y)

    # ------------------------ neighbor compression
    K = 16
    nclasses = 5
    # labels = torch.randint(nclasses, size=(N,))
    # labels = _to_np(torch.randint(nclasses, size=(N,)))
    labels = np.random.randint(nclasses, size=(N,))

    # C, W, Z = protonn(X, labels, K, niters=niters)  # significantly worse
    C, centroid_labels = stochastic_neighbor_compression(X, labels, K, niters=niters)
    # C, centroid_labels = neighbor_compression(X, labels, K, niters=niters)
    print("centroid_labels:", centroid_labels)
    print("C type, shape", type(C), C.shape)
    print("done")


if __name__ == '__main__':
    main()
