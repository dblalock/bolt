#!#!/bin/env/python

from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .utils import kmeans


def neighbor_compression(X, labels, k, niters=1000, verbose=1):
    N, D = X.shape

    # one-hot encode labels
    nclasses = len(np.unique(labels))
    Y = np.zeros((N, nclasses), dtype=np.float32)
    for i in range(N):
        Y[i, labels[i]] = 1

    # intialize centroids
    C, assignments = kmeans(X, k)

    # convert to torch tensors for optimization
    Y = torch.from_numpy(Y)
    C = torch.tensor(C.T, requires_grad=True)  # not from_numpy to allow grad
    X = torch.from_numpy(X)

    loss_fn = torch.nn.CrossEntropyLoss()
    params = [C]
    opt = optim.SGD(params, lr=.1, momentum=.9)

    for t in range(niters):
        temperature = np.log2(t + 2)  # +2 so that it starts at 1 at t=0

        # compute distances to all centroids
        prods = torch.mm(X, C)
        norms_sq = torch.sqrt(torch.sum(C * C))
        dists_sq = prods - norms_sq
        neg_dists_sq = -dists_sq

        # update soft labels for each centroid
        similarities = F.softmax(neg_dists_sq, dim=0)  # N x C; sim to each sample
        class_affinities = similarities.transpose(0, 1) @ Y  # C x nclasses
        class_affinities = F.softmax(class_affinities * temperature, dim=1)

        # update class assignments for inputs
        centroid_similarities = F.softmax(neg_dists_sq * temperature, dim=1)  # N x C
        logits = centroid_similarities @ class_affinities

        # update params and print how we're doing
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (verbose > 0) and ((t + 1) % 10 == 0):
            _, labels_hat = torch.max(logits, dim=1)
            acc = torch.mean((labels == labels_hat).type(torch.float))
            print("acc: ", acc)
            print("{:.3f}".format(loss.item()))  # convert to python float

    return C.cpu().detach().numpy().T, class_affinities.cpu().detach().numpy()


def linear_regression_log_loss(
        X, Y, lamda=1, max_niters=10000, rel_tol=.0001, verbose=2):
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

    return W.cpu().detach().numpy()


def main():
    N, D = 100000, 20
    X = np.random.randn(N, D).astype(np.float32)

    # ------------------------ linear regression with weird loss
    M = 20
    Y = np.random.randn(N, M).astype(np.float32)
    linear_regression_log_loss(X, Y)

    # ------------------------ neighbor compression
    # K = 16
    # nclasses = 5
    # labels = torch.randint(nclasses, size=(N,))

    # C, affinities = neighbor_compression(X, labels, K)
    # print("C type, shape", type(C), C.shape)
    # print("done")


if __name__ == '__main__':
    main()
