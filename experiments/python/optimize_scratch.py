#!#!/bin/env/python

from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .utils import kmeans


def neighbor_compression(X, labels, k, verbose=1):
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

    for t in range(2000):
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
        if (verbose > 0) and (t % 10 == 0):
            _, labels_hat = torch.max(logits, dim=1)
            acc = torch.mean((labels == labels_hat).type(torch.float))
            print("acc: ", acc)
            print("{:.3f}".format(loss.item()))  # convert to python float

    return C.cpu().detach().numpy().T


def main():
    N, D = 10000, 10
    K = 16
    nclasses = 5
    labels = torch.randint(nclasses, size=(N,))
    X = np.random.randn(N, D).astype(np.float32)

    C = neighbor_compression(X, labels, K)
    print("C type, shape", type(C), C.shape)
    print("done")


if __name__ == '__main__':
    main()
