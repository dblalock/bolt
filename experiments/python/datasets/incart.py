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


DATA_DIR = paths.INCART_ECG
NUM_RECORDINGS = 75
NUM_CHANNELS = 12
RAW_DTYPE = np.int16

SAMPLES_PER_SEC = 257
SAMPLES_PER_MIN = SAMPLES_PER_SEC * 60
SAMPLES_PER_HOUR = SAMPLES_PER_MIN * 60


@_memory.cache
def load_recording_ids():
    fpaths = files.list_files(DATA_DIR, abs_paths=False, endswith='.dat')
    assert len(fpaths) == NUM_RECORDINGS
    return fpaths


@_memory.cache
def load_recording(rec_id, limit_nhours=None, dtype=np.float32):
    path = os.path.join(DATA_DIR, rec_id)
    a = np.fromfile(path, dtype=RAW_DTYPE)
    assert len(a) % NUM_CHANNELS == 0
    a = a.reshape(-1, NUM_CHANNELS)  # yep, clearly rowmajor when plotted

    if limit_nhours and limit_nhours > 0:
        a = a[:int(limit_nhours * SAMPLES_PER_HOUR)]
    a = a[SAMPLES_PER_MIN:]  # often a bunch of garbage at the beginning
    a = a.astype(dtype)

    a -= a.mean(axis=0)  # just so r_sq values are more meaningful

    # small amount of smoothing since heavily oversampled + noisy
    # filt = np.hamming(5).astype(np.float32)
    # filt = np.hamming(5).astype(np.float32)
    # filt /= np.sum(filt)
    # for j in range(a.shape[1]):
    #     a[:, j] = np.convolve(a[:, j], filt, mode='same')

    return a


def load_recordings(plot=False, **kwargs):
    rec_ids = load_recording_ids()
    recs = []
    for i, rec_id in enumerate(rec_ids):
        print("loading rec id: ", rec_id)
        rec = load_recording(rec_id, **kwargs)
        recs.append(rec)

        if plot:
            if i < 5:
                offset = 0
                a = rec[offset:(offset + 1000)]
                print("plotting recording {} with shape: {}".format(
                      rec_id, rec.shape))
                plt.figure(figsize=(9, 7))
                plt.plot(a)
                plt.show()
            else:
                return

    return recs


if __name__ == '__main__':
    print("about to call load_recordings")
    load_recordings(plot=True)
    print("called load_recordings")
