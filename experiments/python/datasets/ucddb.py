#!/bin/env python

from __future__ import division, print_function

import numpy as np
# import pyedflib as edf  # pip install pyedflib
# import mne

from . import paths
from . import files

ECG_DIR = paths.UCD_ECG
NUM_RECORDINGS = 25


def main():
    pass
    print("ecg dir: ", ECG_DIR)
    fpaths = files.list_files(ECG_DIR, abs_paths=True)
    # fpaths = files.list_files(ECG_DIR)
    assert len(fpaths) == NUM_RECORDINGS
    # print("fpaths: ", "\n".join(fpaths))
    # print("number of fpaths: ", len(fpaths))

    for path in fpaths:
        print("------------------------ ", path)
        # f = edf.EdfReader(path)
        # print(f.signals_in_file)

        magical_start_offset = 1025  # from looking at raw binary
        # raw = bytes(open(path, 'rb').read())[magical_start_offset:]
        with open(path, 'rb') as f:
            raw = f.read()
        # raw = open(path, 'rb').read()
        print("length of raw: ", len(raw))
        print("type(raw)", type(raw))
        a = np.frombuffer(raw, offset=magical_start_offset, dtype=np.uint16)
        # a = np.frombuffer(raw, dtype=np.uint16)
        print(len(a))
        print(len(a) / 3)
        # print("number of bytes: ", len(raw))
        # with open(path, 'rb') as f:
        #     # f.seek(magical_start_offset)
        #     f.read(magical_start_offset)
        #     a = np.fromfile(f, dtype=np.int16)
        #     print(len(a))
        #     print(len(a) / 3)


if __name__ == '__main__':
    main()
