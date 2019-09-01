#!/bin/env/python

import numpy as np


def energy(A):
    if A.ndim < 2 or len(A) < 2:
        return 0
    diffs = A - A.mean(axis=0)
    return np.sum(diffs * diffs)


def run_trial(N=100, D=3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    w0, w = np.random.randn(2, D)

    X = np.random.randn(N, D)
    X1 = X[(X @ w) > 0]
    X2 = X[(X @ w) <= 0]

    U = X[(X @ w0) > 0]
    V = X[(X @ w0) <= 0]
    U1 = U[(U @ w) > 0]
    U2 = U[(U @ w) <= 0]
    V1 = V[(V @ w) > 0]
    V2 = V[(V @ w) <= 0]

    energy_0 = energy(X)
    energy_w = energy(X1) + energy(X2)

    energy_w0 = energy(U) + energy(V)
    energy_w0_w = energy(U1) + energy(U2) + energy(V1) + energy(V2)

    gain1 = energy_0 - energy_w
    gain2 = energy_w0 - energy_w0_w

    if gain1 < gain2:
        print("N, D, seed = ", N, D, seed)
        print("energy_0:", energy_0)
        print("energy_w:", energy_w)
        print("energy_w0:", energy_w0)
        print("energy_w0_w:", energy_w0_w)
        print("gain1:", gain1)
        print("gain2:", gain2)
        print("w0:\n", w0)
        print("w: \n", w)
        # print("X\t({:.3f}):\n{}".format(energy(X), X))
        # print("X1\t({:.3f}):\n{}".format(energy(X1), X1))
        # print("X2\t({:.3f}):\n{}".format(energy(X2), X2))
        # print("U\t({:.3f}):\n{}".format(energy(U), U))
        # print("U1\t({:.3f}):\n{}".format(energy(U1), U1))
        # print("U2\t({:.3f}):\n{}".format(energy(U2), U2))
        # print("V\t({:.3f}):\n{}".format(energy(V), V))
        # print("V1\t({:.3f}):\n{}".format(energy(V1), V1))
        # print("V2\t({:.3f}):\n{}".format(energy(V2), V2))
        print("X  energy: \t{:.3f}".format(energy(X)))
        print("X1 energy: \t{:.3f}".format(energy(X1)))
        print("X2 energy: \t{:.3f}".format(energy(X2)))
        print("U  energy: \t{:.3f}".format(energy(U)))
        print("U1 energy: \t{:.3f}".format(energy(U1)))
        print("U2 energy: \t{:.3f}".format(energy(U2)))
        print("V  energy: \t{:.3f}".format(energy(V)))
        print("V1 energy: \t{:.3f}".format(energy(V1)))
        print("V2 energy: \t{:.3f}".format(energy(V2)))

        if D == 2:
            import matplotlib.pyplot as plt
            _, axes = plt.subplots(2, 2, figsize=(7.5, 7))
            # plt.scatter(X[:, 0], X[:, 1])
            for ax in axes.ravel():
                ax.set_xlim([-2.5, 2.5])
                ax.set_ylim([-2.5, 2.5])
            #     ax.plot([0, w0[0]], [0, w0[1]])
            #     ax.plot([0, w[0]], [0, w[1]])

            axes[0, 0].set_title("X")
            axes[0, 0].scatter(X[:, 0], X[:, 1])

            axes[0, 1].set_title("U and V (split on w0)")
            axes[0, 1].plot([0, w0[0]], [0, w0[1]])
            axes[0, 1].scatter(U[:, 0], U[:, 1])
            axes[0, 1].scatter(V[:, 0], V[:, 1])

            axes[1, 0].set_title("X1 and X2 (split on w)")
            axes[1, 0].plot([0, w[0]], [0, w[1]])
            axes[1, 0].scatter(X1[:, 0], X1[:, 1])
            axes[1, 0].scatter(X2[:, 0], X2[:, 1])

            axes[1, 1].set_title("U1, U2, V1, V2 (split on w0 and w)")
            axes[1, 1].plot([0, w0[0]], [0, w0[1]])
            axes[1, 1].plot([0, w[0]], [0, w[1]])
            axes[1, 1].scatter(U1[:, 0], U1[:, 1])
            axes[1, 1].scatter(U2[:, 0], U2[:, 1])
            axes[1, 1].scatter(V1[:, 0], V1[:, 1])
            axes[1, 1].scatter(V2[:, 0], V2[:, 1])

            plt.tight_layout()
            plt.show()

        assert gain1 >= gain2


def main():
    ntrials = 100
    # for N in [4, 8, 16, 32, 64, 128, 256]:
    for N in [64, 128, 256]:
        # for D in [1, 2, 3, 5, 10, 100]:
        for D in [100, 200]:
            for trial in range(ntrials):
                run_trial(N=N, D=D, seed=trial)


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    main()
