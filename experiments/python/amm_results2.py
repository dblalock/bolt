#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import pandas as pd
import pprint
from io import StringIO

RESULTS_DIR = os.path.join('results', 'amm')
TIMING_RESULTS_DIR = os.path.join(RESULTS_DIR, 'timing')


def _read_csv_with_garbage(path, **kwargs):
    with open(path, 'r') as f:
        # print("\n".join(f.readlines()))
        keep_lines = [line.strip() for line in f.readlines() if
                      (',' in line and not line.startswith('-'))]
        contents = '\n'.join(keep_lines)
        # print("contents\n", contents)
        return pd.read_csv(StringIO(contents), **kwargs)


def encode_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'encode-timing.csv')
    # ORIG_HEADERS = 'algo __ N D C ___ t0 _0 t1 _1 t2 _2 t3 _3 t4 _4'.split()
    # USE_HEADERS = 'algo N D C t0 t1 t2 t3 t4'.split()
    ORIG_HEADERS = 'algo __ N D C ___ t0 _0 t1 _1 t2 _2'.split()
    USE_HEADERS = 'algo N D C t0 t1 t2'.split()

    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    return df

    # print(df)


def lut_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'lut-timing.csv')
    # ORIG_HEADERS = 'algo __ N D C ___ t0 _0 t1 _1 t2 _2 t3 _3 t4 _4'.split()
    # USE_HEADERS = 'algo N D C t0 t1 t2 t3 t4'.split()
    ORIG_HEADERS = 'algo __ N D C lutconst ___ t0 _0 t1 _1 t2 _2'.split()
    USE_HEADERS = 'algo N D C lutconst t0 t1 t2'.split()

    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    return df


def scan_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'scan-timing.csv')
    ORIG_HEADERS = 'algo __ N C B M ___ t0 _0 t1 _1 t2 _2 t3 _3 t4 _4'.split()
    USE_HEADERS = 'algo N C B M t0 t1 t2 t3 t4'.split()

    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None,
                                skiprows=1)
    df = df[USE_HEADERS]
    return df


def mithral_amm_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'amm-mithral-timing.csv')
    ORIG_HEADERS = ('dset dtype algo __ N D M C lutconst ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'dset dtype algo N D M C lutconst t0 t1 t2 t3 t4'.split()
    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    return df


def bolt_amm_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'amm-bolt-timing.csv')
    ORIG_HEADERS = ('dset dtype algo __ N D M C ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'dset dtype algo N D M C t0 t1 t2 t3 t4'.split()
    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    return df


def dense_amm_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'amm-dense-timing.csv')
    ORIG_HEADERS = ('dset algo __ N D M d ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'dset algo N D M d t0 t1 t2 t3 t4'.split()
    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    return df


def osnap_amm_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'amm-osnap-timing.csv')
    ORIG_HEADERS = ('dset algo __ N D M d s ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'dset algo N D M d s t0 t1 t2 t3 t4'.split()
    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    return df


def sparse_amm_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'amm-sparse-timing.csv')
    ORIG_HEADERS = ('dset algo __ N D M d frac ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'dset algo N D M d frac t0 t1 t2 t3 t4'.split()
    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    return df


def cifar10_amm():
    pass


def cifar100_amm():
    pass


def ucr_amm():
    pass


def caltech_amm():
    pass


def main():
    # print(encode_timings())
    # print(lut_timings())
    # print(scan_timings())
    # print(bolt_amm_timings())
    # print(mithral_amm_timings())
    # print(dense_amm_timings())
    # print(osnap_amm_timings())
    print(sparse_amm_timings())


if __name__ == '__main__':
    main()
