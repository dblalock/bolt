#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import pandas as pd
import pprint
from io import StringIO

RESULTS_DIR = os.path.join('results', 'amm')
TIMING_RESULTS_DIR = os.path.join(RESULTS_DIR, 'timing')

# we log these, but don't need them for the plots
AMM_DROP_COLS = ['__pyience_timestamp__', 'y_mean', 'y_std', 'bias',
                 'raw_mse', 'r', 'alpha', 'ncentroids']


def _read_csv_with_garbage(path, **kwargs):
    with open(path, 'r') as f:
        # print("\n".join(f.readlines()))
        keep_lines = [line.strip() for line in f.readlines() if
                      (',' in line and not line.startswith('-'))]
        contents = '\n'.join(keep_lines)
        # print("contents\n", contents)
        return pd.read_csv(StringIO(contents), **kwargs)


def rename_values_in_col(df, col, name_map, drop_others=True):
    vals = [name_map.get(name.strip().lower(), "") for name in df[col]]
    valid_vals = set(name_map.values())
    # print("valid_vals: ", valid_vals)
    valid_mask = np.array([val in valid_vals for val in vals])
    # print("valid mask: ", valid_mask)
    df[col] = vals
    if drop_others:
        df = df.loc[valid_mask]
    return df
    # print(df)


def melt_observation_cols(df, cols, var_name=None, value_name=None):
    """like pd.melt, but assumes only 1 observation var instead of 1 id var"""
    independent_vars = [col for col in df.columns
                        if col not in set(cols)]
    var_name = 'trial'
    return pd.melt(df, id_vars=independent_vars, value_vars=cols,
                   var_name=var_name, value_name='time')


def melt_times(df, ntimes=5):
    observation_vars = 't0 t1 t2 t3 t4'.split()
    observation_vars = observation_vars[:ntimes]
    return melt_observation_cols(
        df, observation_vars, var_name='trial', value_name='time')


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
    ORIG_HEADERS = ('algo __ N D C B lutconst ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'algo N D C B lutconst t0 t1 t2 t3 t4'.split()

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
    df['fixedB'] = df['algo'].str.strip().str.endswith('noenc')
    df.drop('algo', axis=1, inplace=True)

    return df


def dense_amm_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'amm-dense-timing.csv')
    ORIG_HEADERS = ('dset algo __ N D M d ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'dset algo N D M d t0 t1 t2 t3 t4'.split()
    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    df['algo'] = df['algo'].str.strip()

    # drop stuff that doesn't have fixedW; we just let the existing methods
    # use fixedW (same as fixedB in amm.py), instead of having to encode the
    # smaller matrix
    # df = df.loc[~df['algo'].isin(['blas sketch matmul', 'our sketch matmul'])]

    t_sums = (df['t0'] + df['t1'] + df['t2'] + df['t3'] + df['t4']).values / 5
    # df['t_avg'] = (df['t0'] + df['t1'] + df['t2'] + df['t3'] + df['t4']) / 5.

    # # mark whether it's from our gemm or eigen gemm
    # df['is_ours'] = df['algo'].str.startswith('our')

    print("uniq n vals: ", np.unique(df['N']))

    sizes = np.empty((len(df), 4), dtype=np.int)
    sizes[:, 0] = df['N']
    sizes[:, 1] = df['D']
    sizes[:, 2] = df['M']
    sizes[:, 3] = df['d']
    as_tuples = [tuple(row) for row in sizes]
    uniq_tuples = sorted(list(set(as_tuples)))
    keep_idxs = []
    # print("sizes:\n", sizes)
    # print("uniq_tuples:\n", uniq_tuples)
    for tup in uniq_tuples:
        row = np.array(tup)
        idxs = np.where((sizes == row).sum(axis=1) == sizes.shape[1])[0]
        best_idx = idxs[np.argmin(t_sums[idxs])]
        print(f"{tup} -> {best_idx}")
        keep_idxs.append(best_idx)
    df = df.iloc[keep_idxs]

    rename_dict = {}
    rename_dict['blas matmul'] = 'Brute Force'
    rename_dict['our matmul'] = 'Brute Force'
    rename_dict['blas sketch matmul'] = 'Dense Sketch'
    rename_dict['our sketch matmul'] = 'Dense Sketch'
    rename_dict['blas sketch fixedw matmul'] = 'Dense Sketch'
    rename_dict['our sketch fixedw matmul'] = 'Dense Sketch'
    df = rename_values_in_col(df, 'algo', rename_dict, drop_others=False)

    return df


def osnap_amm_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'amm-osnap-timing.csv')
    ORIG_HEADERS = ('dset algo __ N D M d s ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'dset algo N D M d s t0 t1 t2 t3 t4'.split()
    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    df.drop('algo', axis=1, inplace=True)
    return df


def sparse_amm_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'amm-sparse-timing.csv')
    ORIG_HEADERS = ('dset algo __ N D M d frac ___ '
                    't0 _0 t1 _1 t2 _2 t3 _3 t4 _4').split()
    USE_HEADERS = 'dset algo N D M d frac t0 t1 t2 t3 t4'.split()
    df = _read_csv_with_garbage(TIMINGS_PATH, names=ORIG_HEADERS, header=None)
    df = df[USE_HEADERS]
    df.drop('algo', axis=1, inplace=True)
    return df


# def _extract_cols_into_list_of_tuples(df, cols):
def _extract_cols_into_list_of_tuples(df, cols):
    # return [tuple(row) for row in df[cols].iterrows()]
    ar = np.vstack([df[col] for col in cols]).T
    # print("ar: \n", ar)
    ar = np.atleast_2d(ar).astype(np.int)
    # return [tuple(row) for row in ar]
    return [sum([(i + 1) * hash(val)
            for i, val in enumerate(row)]) for row in ar]
    # return [int(hash(tuple(row))) for row in ar]


def _join_on_cols(df_left, left_cols, df_right, right_cols):
    df_left['__index__'] = _extract_cols_into_list_of_tuples(
        df_left, left_cols)
    df_right['__index__'] = _extract_cols_into_list_of_tuples(
        df_right, right_cols)

    dup_cols = set(left_cols) & set(right_cols)
    print("dup cols: ", dup_cols)
    df_right = df_right.drop(dup_cols, axis=1)

    df = df_left.merge(df_right, on='__index__', how='left')
    df.drop(['__index__'], axis=1, inplace=True)
    return df


def _join_with_bolt_times(df):
    time_df = bolt_amm_timings()
    df = df.loc[df['method'].str.lower().str.startswith('bolt')]
    return _join_on_cols(df, 'N D M ncodebooks'.split(),
                         time_df, 'N D M C'.split())


def _join_with_osnap_times(df):
    time_df = osnap_amm_timings()
    df = df.loc[df['method'].str.lower().str.startswith('osnap')]
    return _join_on_cols(df, 'N D M d'.split(),
                         time_df, 'N D M d'.split())


def _join_with_brute_force_times(df):
    time_df = dense_amm_timings()
    df = df.loc[df['method'].str.lower().str.startswith('exact')]
    time_df = time_df.loc[time_df['algo'].str.lower().str.startswith('brute')]
    # print(df)
    print(time_df)




    # TODO pick up here




def cifar10_amm():
    RESULTS_PATH = os.path.join(RESULTS_DIR, 'cifar10.csv')
    df = pd.read_csv(RESULTS_PATH)
    df.drop(AMM_DROP_COLS + ['task_id'], axis=1, inplace=True)  # only 1 task
    # print(df)

    # # df = _join_with_times(df)
    # df_bolt = _join_with_bolt_times(df)
    # df_osnap = _join_with_osnap_times(df)
    df_brute = _join_with_brute_force_times(df)

    # return df_osnap
    return df_brute


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
    # print(sparse_amm_timings())
    print(cifar10_amm())


if __name__ == '__main__':
    main()
