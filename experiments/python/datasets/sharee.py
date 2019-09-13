#!/bin/env python

from __future__ import division, print_function

import numpy as np

from . import paths
from . import files

DATA_DIR = paths.SHAREE_ECG
NUM_RECORDINGS = 139


def load_recordings():
    fpaths = files.list_files(DATA_DIR, abs_paths=True, endswith='.dat')
    # print("len fpaths: ", len(fpaths))
    assert len(fpaths) == NUM_RECORDINGS

    # for path in fpaths[:5]:
    for path in fpaths:
        # print("------------------------ ", path)

        # with open(path, 'rb') as f:
        #     raw = f.read()
        # print("length of raw: ", len(raw))
        # print("type(raw)", type(raw))

        # a = np.frombuffer(raw, dtype=np.uint16)
        a = np.fromfile(path, dtype=np.uint16)
        # print(len(a))
        # print(len(a) / 3)
        assert len(a) % 3 == 0
        a = a.reshape(-1, 3)  # looks like it's rowmajor

        yield a

        # recordings.append(a.reshape(-1, 3))

        # import matplotlib.pyplot as plt
        #
        # offset = int(1e6)
        # a = a[offset:(offset + 1000)]
        # plt.plot(a)
        # plt.show()


if __name__ == '__main__':
    load_recordings()
