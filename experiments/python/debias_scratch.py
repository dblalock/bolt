#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def main():
    UNDEFINED = 7

    M = 40000
    # M = 500
    # M = 2
    # K = 16
    # C = 64

    try_Cs = np.array([2, 4, 8, 16, 32, 64, 128])
    try_Us = np.array([2, 4, 8, 16, 32, 64, 128])

    biases = np.zeros((try_Cs.size, try_Us.size)) + UNDEFINED
    # sses = np.zeros((try_Cs.size, try_Us.size)) + UNDEFINED
    dists_true = np.zeros((try_Cs.size, try_Us.size, M))
    dists_hat = np.zeros((try_Cs.size, try_Us.size, M))
    all_errs = np.zeros((try_Cs.size, try_Us.size, M))
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

            errs = dists - true_dists
            # biases[i, j] = diffs.mean()
            biases[i, j] = errs.mean()

            # store true dists so we can compute variance of estimator
            # dists_true[i, j] = true_dists
            # dists_hat[i, j] = dists
            all_errs[i, j] = errs

            # debias =
            # sses[i, j] = diffs

            # diffs = true_dists - dists
            # print(f"C = {C}, upcast_every={upcast_every}")
            # print("mean true dist: ", true_dists.mean())
            # print("mean diff:", diffs.mean())

    print("biases:\n", biases)

    # col = try_Cs / 4
    # row = np.log2(try_Us).astype(np.int)
    # biases_hat = np.outer(col, row)
    # print("biases_hat:\n", biases_hat)

    # biases_hat2 = np.zeros((try_Cs.size, try_Us.size)) - UNDEFINED
    biases_hat2 = np.zeros((try_Cs.size, try_Us.size))
    for i, C in enumerate(try_Cs):
        for j, upcast_every in enumerate(try_Us):
            if upcast_every > C:
                continue
            biases_hat2[i, j] = C / 4 * np.log2(upcast_every)
    print("biases_hat2:\n", biases_hat2)

    print("corrected biases:\n", biases - biases_hat2)

    all_errs -= biases_hat2[..., np.newaxis]
    # print("mean corrected errs:\n", all_errs.mean(axis=-1))
    print("mean corrected errs:\n", np.var(all_errs, axis=-1))
    sq_errs = (all_errs * all_errs).mean(axis=-1)
    print("empirical mean squared err for C, U", sq_errs)

    sq_errs_hat = np.zeros((try_Cs.size, try_Us.size))
    for i, C in enumerate(try_Cs):
        for j, upcast_every in enumerate(try_Us):
            if upcast_every > C:
                continue
            sq_errs_hat[i, j] = C / 8 * np.log2(upcast_every)
    print("estimated mean squared err for C, U", sq_errs_hat)

    print("takeaway: no idea what closed form for mse is...")


    # print("biases - biases_hat", biases - biases_hat)


    # plt.scatter(true_dists, dists)
    # plt.show()


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:5.1f}".format(f)},
                        linewidth=100)
    main()
