#!/bin/env python

# Load 3-lead ECG recordings from SHAREE Database:
# https://physionet.org/content/shareedb/1.0.0/

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import os

from . import paths
from . import files

from joblib import Memory
_memory = Memory('.', verbose=0)


DATA_DIR = paths.SHAREE_ECG
NUM_RECORDINGS = 139

SAMPLES_PER_SEC = 128
SAMPLES_PER_MIN = SAMPLES_PER_SEC * 60
SAMPLES_PER_HOUR = SAMPLES_PER_MIN * 60


# @_memory.cache
def load_recording_ids():
    # fpaths = files.list_files(DATA_DIR, abs_paths=True, endswith='.dat')
    fpaths = files.list_files(DATA_DIR, abs_paths=False, endswith='.dat')
    assert len(fpaths) == NUM_RECORDINGS
    return fpaths


def load_recording(rec_id, limit_nhours=None, dtype=np.float32):
    path = os.path.join(DATA_DIR, rec_id)
    a = np.fromfile(path, dtype=np.uint16)
    assert len(a) % 3 == 0
    a = a.reshape(-1, 3)  # looks like it's rowmajor
    # a = a.reshape(3, -1).T  # is colmajor clearly worse? EDIT: yes

    if limit_nhours and limit_nhours > 0:
        a = a[:limit_nhours * SAMPLES_PER_HOUR]
    a = a[10*SAMPLES_PER_MIN:]  # often a bunch of garbage at the beginning

    # small amount of smoothing since heavily oversampled + noisy
    filt = np.hamming(5).astype(dtype)
    filt /= np.sum(filt)
    for j in range(a.shape[1]):
        # print("a shape: ", a[:, j].shape)
        # smoothed = np.convolve(filt, a[:, j], mode='same')
        # print("smoothed shape: ", smoothed.shape)
        # a[:, j] = smoothed
        a[:, j] = np.convolve(a[:, j], filt, mode='same')

    return a.astype(dtype)


# def load_recordings(generator=False, plot=False, **kwargs):
def load_recordings(plot=False, **kwargs):
    rec_ids = load_recording_ids()

    print("actually called load_recordings")
    print("len(rec ids): ", len(rec_ids))
    # import sys; sys.exit()

    recs = []
    for i, rec_id in enumerate(rec_ids):
        print("rec id: ", rec_id)
        rec = load_recording(rec_id, **kwargs)
        recs.append(rec)
        # if generator:
        #     yield rec
        # else:
        #     recs.append(rec)
        recs.append(rec)

        if plot and i < 5:
            offset = SAMPLES_PER_MIN
            a = rec[offset:(offset + 1000)]
            print('about to plot recording', rec_id)
            plt.figure(figsize=(9, 7))
            plt.plot(a)
            plt.show()

    return recs

    # if not generator:
    #     return recs


if __name__ == '__main__':
    # print("done")
    print("about to call load_recordings")
    load_recordings(plot=True)
    # print("rec ids: ", load_recording_ids())
    print("called load_recordings")
