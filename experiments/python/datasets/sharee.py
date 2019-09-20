#!/bin/env python

# Load 3-lead ECG recordings from SHAREE Database:
# https://physionet.org/content/shareedb/1.0.0/

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import os

from . import paths
from . import files

DATA_DIR = paths.SHAREE_ECG
NUM_RECORDINGS = 139


def load_recording_ids():
    # fpaths = files.list_files(DATA_DIR, abs_paths=True, endswith='.dat')
    fpaths = files.list_files(DATA_DIR, abs_paths=False, endswith='.dat')
    assert len(fpaths) == NUM_RECORDINGS
    return fpaths


def load_recording(rec_id):
    path = os.path.join(DATA_DIR, rec_id)
    a = np.fromfile(path, dtype=np.uint16)
    assert len(a) % 3 == 0
    return a.reshape(-1, 3)  # looks like it's rowmajor


def load_recordings(generator=True, plot=False):
    rec_ids = load_recording_ids()

    recs = []
    for rec_id in rec_ids:
        rec = load_recording(rec_id)
        if generator:
            yield rec
        else:
            recs.append(rec)

        if plot:
            offset = int(1e6)
            a = rec[offset:(offset + 1000)]
            print('about to plot recording', rec_id)
            plt.figure(figsize=(9, 7))
            plt.plot(a)
            plt.show()

    if not generator:
        return recs


if __name__ == '__main__':
    print("about to call load_recordings")
    load_recordings()
    # print("rec ids: ", load_recording_ids())
    print("called load_recordings")
