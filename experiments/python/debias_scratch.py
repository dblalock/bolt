#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def main():
    M = 1000
    # M = 500
    # M = 2
    # K = 16
    # C = 64

    try_Cs = np.array([2, 4, 8, 16, 32, 64, 128])
    try_Us = np.array([2, 4, 8, 16, 32, 64, 128])

    biases = np.zeros((try_Cs.size, try_Us.size)) + 7
    for i, C in enumerate(try_Cs):
        for j, upcast_every in enumerate(try_Us):
            if upcast_every > C:
                continue

            # dists = np.random.randint(256, size=(M * K, C))
            orig_dists = np.random.randint(256, size=(M, C))
            # print("orig_dists[:10]", orig_dists[:10])  # ya, these are sane
            dists = orig_dists.reshape(orig_dists.shape[0], -1, upcast_every)

            while dists.shape[-1] > 2:
                # print("dists shape: ", dists.shape)
                # print("dists:\n", dists)
                dists = (dists[:, :, ::2] + dists[:, :, 1::2] + 1) // 2
            # print("dists shape: ", dists.shape)
            dists = (dists[:, :, 0] + dists[:, :, 1] + 1) // 2
            dists = dists.sum(axis=-1)  # clipping not needed
            dists *= upcast_every

            true_dists = orig_dists.sum(axis=1)

            biases[i, j] = (true_dists - dists).mean()

            # diffs = true_dists - dists
            # print(f"C = {C}, upcast_every={upcast_every}")
            # print("mean true dist: ", true_dists.mean())
            # print("mean diff:", diffs.mean())

    print("biases:\n", biases)

    # col = try_Cs / 4
    # row = np.log2(try_Us).astype(np.int)
    # biases_hat = np.outer(col, row)
    # print("biases_hat:\n", biases_hat)

    biases_hat2 = np.zeros((try_Cs.size, try_Us.size)) + 7
    for i, C in enumerate(try_Cs):
        for j, upcast_every in enumerate(try_Us):
            if upcast_every > C:
                continue
            biases_hat2[i, j] = C / 4 * np.log2(upcast_every)
    print("biases_hat2:\n", biases_hat2)

    # print("biases - biases_hat", biases - biases_hat)


    # plt.scatter(true_dists, dists)
    # plt.show()


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:5.1f}".format(f)},
                        linewidth=100)
    main()
