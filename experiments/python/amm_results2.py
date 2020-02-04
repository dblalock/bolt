#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import pandas as pd
from io import StringIO

from . import amm_methods as methods

from joblib import Memory
_memory = Memory('.', verbose=1)

pd.options.mode.chained_assignment = None  # suppress stupid warning


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
    name_map = {k.strip().lower(): v for k, v in name_map.items()}
    vals = [name_map.get(name.strip().lower(), "") for name in df[col]]
    valid_vals = set(name_map.values())
    # print("valid_vals: ", valid_vals)
    valid_mask = np.array([val in valid_vals for val in vals])
    # print("valid mask: ", valid_mask)
    df = df.copy()
    df[col] = vals
    if drop_others:
        df = df.loc[valid_mask]
    return df
    # print(df)


def melt_observation_cols(df, cols, var_name=None, value_name=None):
    """like pd.melt, but assumes only 1 observation var instead of 1 id var"""
    independent_vars = [col for col in df.columns
                        if col not in set(cols)]
    return pd.melt(df, id_vars=independent_vars, value_vars=cols,
                   var_name=var_name, value_name='time')


def melt_times(df, ntimes=5):
    observation_vars = 't0 t1 t2 t3 t4'.split()
    observation_vars = observation_vars[:ntimes]
    return melt_observation_cols(
        df, observation_vars, var_name='timing_trial', value_name='time')


def drop_cols_inplace(df, cols):
    for col in AMM_DROP_COLS:
        try:
            df.drop([col], axis=1, inplace=True)
        except KeyError:
            pass
    return df


def frac_above_thresh(df, xvar, yvar, methodvar, unitvar, ythresh):
    """
    (method, xvar) -> [0, 1]

    Assumes you have a tidy dataframe where method, xvar, yvar, and unit are
    each a col.
    """
    df = df.copy()
    # df['frac_above_thresh'] = (df[yvar] > ythresh).astype(np.float)

    # (method, xvar, [(unit, yvar)]) -> bool
    df['frac_above_thresh'] = (df[yvar] > ythresh).astype(np.float)

    # independent_vars = [methodvar, unitvar, xvar]
    independent_vars = [methodvar, xvar]
    # return df.groupby(independent_vars)['is_above_thresh'].transform('mean')

    # last part converts from groupby back to regular df EDIT: no it doesn't
    # return df.groupby(independent_vars)['frac_above_thresh'].mean().apply(pd.Series)

    # return df.groupby(independent_vars)['is_above_thresh'].mean()
    df = df.groupby(independent_vars)['frac_above_thresh'].mean()
    # this is the magic line; turn multi-index levels into regular cols, with
    # multi-index value broadcast all the corresponding rows;
    # WOW this took a long time to figure out...
    df = df.reset_index(level=independent_vars)
    return df


    # tmp = df.groupby(independent_vars)['frac_above_thresh'].mean()
    # tmp = df.groupby(independent_vars)['frac_above_thresh'].transform('mean')
    # print("tmp:\n", tmp)
    # df['frac_above_thresh'] = tmp
    # return df

    # return df.groupby(independent_vars)[independent_vars + ['frac_above_thresh']]
    # ret = df.groupby([methodvar, unitvar, xvar])['__above_thresh__']
    # ret.drop(['__above_thresh__'], axis=1, inplace=True)
    # return ret


def encode_timings():
    TIMINGS_PATH = os.path.join(TIMING_RESULTS_DIR, 'encode-timing.csv')
    ORIG_HEADERS = 'algo __ N D C B ___ t0 _0 t1 _1 t2 _2 t3 _3 t4 _4'.split()
    USE_HEADERS = 'algo N D C B t0 t1 t2 t3 t4'.split()
    # ORIG_HEADERS = 'algo __ N D C ___ t0 _0 t1 _1 t2 _2'.split()
    # USE_HEADERS = 'algo N D C t0 t1 t2'.split()

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
    df = df.loc[df['fixedB']]

    # print("bolt df:\n", df)
    # import sys; sys.exit()

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

    # print("uniq n vals: ", np.unique(df['N']))

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
        # print(f"{tup} -> {best_idx}")
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
    return [sum([hash(-12435 * i + 1) ^ hash(1234567 * val)
            for i, val in enumerate(row)]) for row in ar]
    # return [int(hash(tuple(row))) for row in ar]


def _join_on_cols(df_left, left_cols, df_right, right_cols, verbose=0):
    df_left = df_left.copy()
    df_right = df_right.copy()
    df_left['__index__'] = _extract_cols_into_list_of_tuples(
        df_left, left_cols)
    df_right['__index__'] = _extract_cols_into_list_of_tuples(
        df_right, right_cols)

    # dup_cols = set(left_cols) & set(right_cols)
    # if verbose > 0:
    #     print("_join_on_cols(); dropping duplicate cols from rhs: ", dup_cols)
    # df_right = df_right.drop(dup_cols, axis=1)

    df = df_left.merge(
        df_right, on='__index__', how='left', suffixes=('', '_rhs'))
    df.drop(['__index__'], axis=1, inplace=True)
    # df.sort_values(left_cols, axis=0, inplace=True)
    return df


def _join_with_mithral_times(df, timing_dtype='f32'):
    time_df = mithral_amm_timings()
    if timing_dtype is not None:
        time_df = time_df.loc[time_df['dtype'].str.strip() == timing_dtype]
    df = df.loc[df['method'].str.lower().str.startswith('mithral')]
    df['ncodebooks'] = df['ncodebooks'].astype(np.int)

    # time_df.reset_index(inplace=True, drop=True)
    # df.reset_index(inplace=True, drop=True)
    # print("time_df with appropriate dtype:\n", time_df)
    # import sys; sys.exit()

    # we also report times for subroutines within mithral; can't let it
    # use any of these; just use rename_values_in_col to drop them and also
    # get more intuitive debug output
    # rename_dict = {'amm mithral sparselut': 'Mithral, L = ??',
    #                'amm mithral nolut': 'Mithral, L = ∞'}
    name_mithral_dense = 'mithralDense'  # only one we use; others arbitrary
    rename_dict = {'amm mithral sparselut': 'mithralSparse',
                   'amm mithral denselut': name_mithral_dense,
                   'amm mithral nolut': 'mithralOffline'}
    time_df = rename_values_in_col(time_df, 'algo', rename_dict)

    # give MithralPQ a valid lut const so the join will work (pq is equivalent
    # to constant of 1)
    is_mithral_pq = df['method'].str.lower().str.startswith('mithralpq')
    df.loc[is_mithral_pq, 'lut_work_const'] = 1
    df_mpq = df.loc[is_mithral_pq].copy()

    # there shouldn't be rows that violated this, but there are (probably
    # from early runs that haven't been overwritten yet)
    df = df.loc[df['lut_work_const'].values <= df['ncodebooks'].values]

    # now add in extra rows for mithral with no lut computation (which is
    # assumed to use dense luts because no reason not to) vs mithral
    # with dense lut computation as part of the timing
    is_any_mithral = df['method'].str.lower().str.startswith('mithral')
    is_mithral = is_any_mithral & (~is_mithral_pq)
    is_dense = df['lut_work_const'] == -1
    df_mithral_dense = df.loc[is_mithral & is_dense].copy()
    dummy_lutconst = -2
    df_mithral_dense['lut_work_const'] = dummy_lutconst
    time_df['lutconst'].loc[
        time_df['algo'] == name_mithral_dense] = dummy_lutconst

    # add in version of mithralpq with offline lut computation
    df_mpq = df.loc[df['method'].str.lower().str.startswith('mithralpq')]
    df_mpq['lut_work_const'] = -1

    df = pd.concat([df, df_mithral_dense, df_mpq], axis=0)

    cols_df = 'N D M ncodebooks lut_work_const'.split()
    cols_time_df = 'N D M C lutconst'.split()

    # print("df cols: ", df.columns)
    # time_df.reset_index(inplace=True, drop=True)
    # df.reset_index(inplace=True, drop=True)
    df.sort_values(['method'] + cols_df, axis=0, inplace=True)
    time_df.sort_values(cols_time_df, axis=0, inplace=True)

    ret = _join_on_cols(df, cols_df, time_df, cols_time_df)
    # ret['lut_work_const'].loc[ret['lut_work_const'] == dummy_lutconst] = -1

    # show_keys = 'method N D M C ncodebooks lutconst lut_work_const'.split()
    # print("mithral df:\n", df['method N D M ncodebooks lut_work_const'.split()])
    # # print("mithral time df:\n", time_df.loc[time_df['dset'] == 'Cifar10'])
    # print("mithral time df:\n", time_df.loc[time_df['dset'] == 'Caltech3x3'])
    # print("joined df:\n", ret[show_keys])
    # # print("joined df:\n", ret)
    # import sys; sys.exit()

    # one of these fails if the join failed; check if you have redundant
    # rows in either df or missing rows in the time df
    assert np.all(ret['C'] == ret['ncodebooks'])
    assert np.all(ret['lutconst'] == ret['lut_work_const'])
    return ret


def _join_with_bolt_times(df):
    time_df = bolt_amm_timings()
    df = df.loc[df['method'].str.lower().str.startswith('bolt')]
    return _join_on_cols(df, 'N D M ncodebooks'.split(),
                         time_df, 'N D M C'.split())


def _join_with_osnap_times(df):
    time_df = osnap_amm_timings()
    # df = df.loc[df['method'].str.lower().str.startswith('osnap')]
    df = df.loc[df['method'].isin(
        [methods.METHOD_OSNAP, methods.METHOD_HASHJL])]
    df['s'] = 1
    df['s'].loc[df['method'] == methods.METHOD_OSNAP] = 4
    # print("osnap df shape: ", df.shape)
    df['d'] = df['d'].astype(np.int)

    # print("time_df:\n", time_df[time_df['dset'] == 'Cifar10'])
    # note that d < s isn't present in time_df, which makes sense
    return _join_on_cols(df, 'N D M d s'.split(),
                         time_df, 'N D M d s'.split())


def _join_with_brute_force_times(df):
    time_df = dense_amm_timings()
    df = df.loc[df['method'].str.lower().str.startswith('exact')]
    time_df = time_df.loc[time_df['algo'].str.lower().str.startswith('brute')]
    # print("df:\n", df)
    # print("time_df:\n", time_df)
    return _join_on_cols(df, 'N D M'.split(), time_df, 'N D M'.split())


def _join_with_dense_sketch_times(df):
    time_df = dense_amm_timings()
    # print("found methods in df: ", df['method'].unique())
    # print("dense sketch methods: ", methods.DENSE_SKETCH_METHODS)
    df = df.loc[df['method'].isin(methods.DENSE_SKETCH_METHODS)]
    time_df = time_df.loc[time_df['algo'].str.lower().str.startswith(
        'dense sketch')]
    # print("df:\n", df)
    # print("time_df:\n", time_df)
    return _join_on_cols(df, 'N D M d'.split(),
                         time_df, 'N D M d'.split())


def extract_pareto_frontier_idxs(xvals, yvals):
    """assumes lower x is better and higher y is better"""
    assert len(xvals) == len(yvals)
    sort_idxs = np.argsort(xvals)
    xvals = xvals[sort_idxs]
    yvals = yvals[sort_idxs]
    # orig_idxs = np.arange(len(xvals))
    first_idx = sort_idxs[0]
    curr_thresh = yvals[first_idx]
    keep_idxs = [first_idx]
    for i, y in enumerate(yvals[1:]):
        if y > curr_thresh:
            curr_thresh = y
            keep_idxs.append(sort_idxs[i + 1])
    return keep_idxs


def _join_with_sparse_sketch_times(df, sparse_pareto=True):
    time_df = sparse_amm_timings()
    df = df.loc[df['method'].str.lower().str.startswith('sparse')]
    df['d'] = df['d'].astype(np.int)

    new_rows = []
    for _, row in df.iterrows():
        # pprint.pprint(dict(row))
        subdf = time_df
        for key in 'N D M d'.split():
            subdf = subdf.loc[subdf[key] == row[key]]
        if len(subdf) < 1:
            continue
        sparsities = subdf['frac']
        # print("subdf for N, D, M, D: ", [row[k] for k in 'N D M d'.split()])
        # print(subdf)
        # N, D, M, d = [row[k] for k in 'N D M d'.split()]
        target_frac = row['sparsity']
        small_enough_sparsities_idxs = np.where(sparsities.values <= target_frac)[0]
        if len(small_enough_sparsities_idxs):
            take_idx = small_enough_sparsities_idxs[-1]
        else:  # no nonzeros, or at least uselessly few of them
            take_idx = np.argmin(sparsities.values)

        time_keys = 't0 t1 t2 t3 t4'.split()
        times_row = subdf.iloc[take_idx]
        # times = subdf.loc[take_idx, time_keys]
        row = dict(row)
        for key in time_keys:
            row[key] = float(times_row[key])
        row['time'] = sum([float(times_row[key])
                          for key in time_keys]) / len(time_keys)

        new_rows.append(row)
    # return pd.DataFrame.from_records(new_rows)
    df = pd.DataFrame.from_records(new_rows)

    if not sparse_pareto:
        return df

    # # for dset in df['']

    # subdf = df.loc[df['method'] == 'SparsePCA']

    # here we have a bunch of hack stuff
    # print("df columns: ", df.columns)
    # yvals = 1. - df['normalized_mse'].values

    subdfs = []
    for tid in df['task_id'].unique():
        subdf = df.loc[df['task_id'] == tid]
        xvals = subdf['time'].values
        if 'acc_amm' in df.columns:
            yvals = subdf['acc_amm'].values
        else:
            yvals = 1. - subdf['normalized_mse'].values
        idxs = extract_pareto_frontier_idxs(xvals, yvals)
        subdfs.append(subdf.iloc[idxs])
    df = pd.concat(subdfs, axis=0)

    return df


def _clean_method_names_amm(df):
    key = 'method' if 'method' in df else 'algo'
    if 'lutconst' in df:

        df.loc[df['lutconst'] == -2, key] = 'MADDNESS Dense'
        is_lutconst_neg1 = df['lutconst'] == -1
        is_mithral_pq = df['method'] == 'MithralPQ'
        df.loc[is_lutconst_neg1 & is_mithral_pq, key] = 'MADDNESS-PQ'
        df.loc[is_lutconst_neg1 & ~is_mithral_pq, key] = 'MADDNESS'
        df.loc[df['lutconst'] == 1, key] = 'MADDNESS, L = 1'
        df.loc[df['lutconst'] == 2, key] = 'MADDNESS, L = 2'
        df.loc[df['lutconst'] == 4, key] = 'MADDNESS, L = 4'

        # df.loc[df['lutconst'] == -2, key] = 'Mithral Dense'
        # is_lutconst_neg1 = df['lutconst'] == -1
        # is_mithral_pq = df['method'] == 'MithralPQ'
        # df.loc[is_lutconst_neg1 & is_mithral_pq, key] = 'MithralPQ'
        # df.loc[is_lutconst_neg1 & ~is_mithral_pq, key] = 'Mithral'
        # df.loc[df['lutconst'] == 1, key] = 'Mithral, L = 1'
        # df.loc[df['lutconst'] == 2, key] = 'Mithral, L = 2'
        # df.loc[df['lutconst'] == 4, key] = 'Mithral, L = 4'

        # mask = df['lutconst'] == 1
        # is_mithral_pq = df[key].str.lower().str.startswith('mithralpq')
        # mask &= ~is_mithral_pq
        # df[key][mask] = 'Mithral, L = ∞'
    # df[key].loc[df[key] == 'Exact'] = 'Brute Force'

    df[key].loc[df[key] == 'Exact'] = 'Exact'

    return df


def _clean_metrics_amm(df):
    df = df.rename({'acc_amm': 'Accuracy'}, axis=1)
    df['time'] = (df['t0'] + df['t1'] + df['t2'] + df['t3'] + df['t4']) / 5.
    df['Throughput'] = 1e3 * df['N'] * df['M'] / df['time']

    # create ops column that sums number of multiplies + lookups
    df['muls'] = df['muls'].fillna(0)
    mask = ~df['nlookups'].isna()
    df['ops'] = df['muls']
    # print("debugging df[ops]: ")
    # # print(type(df['ops']))
    # # print(type(df['ops']))
    # print(type(df['nlookups']))
    df['ops'].loc[mask] += df['nlookups'].loc[mask]

    # df['nor']
    # df_exact = df.loc[df['method'] == 'Brute Force']
    df_exact = df.loc[df['method'] == 'Exact']
    # print("df_exact\n", df_exact)
    if 'task_id' in df.columns:
        nuniq_tasks = len(df['task_id'].unique())
    else:
        nuniq_tasks = 1  # cifar{10,100}
    assert df_exact.shape[0] == nuniq_tasks
    base_time = float(df_exact.loc[0, 'time'])
    df['NormalizedTime'] = df['time'] / base_time
    df['Speedup'] = 1. / df['NormalizedTime']
    df['1 - NMSE'] = 1. - df['normalized_mse']
    if 'Accuracy' in df.columns:
        df['Relative Accuracy'] = df['Accuracy'] / (df['acc_orig'] + 1e-20)
        # print("df.columns", df.columns)
        # df['Change in Accuracy'] = df['Accuracy'] - df['acc-1nn-raw']
        # if 'acc-1nn-raw' in df.columns:
        # # # note that relative accuracy can actually be higher if errors
        # # # happen to compensate for incorrect classification sometimes
        # # print("max relative acc: ", df['Relative Accuracy'].values.max())
        # # # assert df['Relative Accuracy'].values.max() <= 1.000001

        # # acc_orig field is supposed to capture this, but I messed it up for
        # # 1nn so this will also work
        # tid2acc = {}
        # exactdf = df.loc[df['method'] == 'Exact']
        # for tid in df['task_id'].unique():
        #     subdf = exactdf.loc[exactdf['task_id'] == tid]
        #     tid2acc[tid] = subdf['Accuracy'].values[0]
        # df['BaseAccuracy'] = [tid2acc[tid] for tid in df['task_id']]
        # df['Relative Accuracy'] = df['Accuracy'] / df['BaseAccuracy']

    return df


def _join_with_times(df, timing_dtype='f32', sparse_pareto=True):

    df_bolt = _join_with_bolt_times(df)

    # # print("df bolt:\n", df_bolt)
    # df_tmp = df_bolt['N D M C ncodebooks method normalized_mse t0 t1 t2 t3 t4 task_id'.split()]
    # df_tmp = df_tmp.loc[df_tmp['task_id'].isin(['ucr Yoga k=128', 'ucr Wafer k=128'])]

    # df_tmp['time'] = (df_tmp['t0'] + df_tmp['t1'] + df_tmp['t2'] + df_tmp['t3'] + df_tmp['t4']) / 5.

    # print("df tmp:\n", df_tmp)
    # print("df tmp times:\n", df_tmp[['C', 'time', 'task_id']])

    # # tids = df_tmp['task_id'].unique()

    # # # yep, exactly 6 results per dset
    # # counts = df_tmp.groupby('task_id')['task_id'].count()
    # # print("task counts: ", counts)

    # import sys; sys.exit()

    # print("df bolt:\n", df_bolt)  # looks good
    # import sys; sys.exit()

    # assert np.all(df_mithral['lutconst'] == df_mithral['lut_work_const'])

    # df_mithral = df.loc[df['method'].str.startswith('Mithral')]
    # df_mithral.to_csv('mithral-caltech-debug.csv')

    df_mithral = _join_with_mithral_times(df, timing_dtype=timing_dtype)
    # df_tmp = df_mithral
    # df_tmp = df_tmp['N D M C ncodebooks lutconst lut_work_const method algo normalized_mse t0 t1'.split()]
    # # print("mithral rows:\n", df.loc[df['method'].str.startswith('mithral')])
    # print("mithralpq rows after join:\n", df_tmp.loc[df_tmp['method'] == 'MithralPQ'])
    # print("mithral rows after join:\n", df_tmp[:100])
    # mismatch_mask = df_tmp['lutconst'] != df_tmp['lut_work_const']
    # print("mithral mismatched rows:\n", df_tmp.loc[mismatch_mask])
    # print(df_mithral['lutconst', 'lut_work_const'])
    # import sys; sys.exit()

    # if this line fails, it's usually because the join with mithral times
    # failed
    assert np.all(df_mithral['lutconst'] == df_mithral['lut_work_const'])

    df_osnap = _join_with_osnap_times(df)
    df_brute = _join_with_brute_force_times(df)
    df_sketch = _join_with_dense_sketch_times(df)
    df_sparse = _join_with_sparse_sketch_times(df, sparse_pareto=sparse_pareto)
    dfs = [df_mithral, df_bolt, df_osnap, df_brute, df_sketch, df_sparse]

    return pd.concat(dfs, axis=0, join='outer', sort=False)


def _clean_amm_results_df(df, timing_dtype='f32', sparse_pareto=True):
    # print("initial methods: ", df['method'].unique())
    df = _join_with_times(
        df, timing_dtype=timing_dtype, sparse_pareto=sparse_pareto)
    # df['time'] = df['t_avg']
    # df = melt_times(df)

    df = _clean_metrics_amm(df)
    df = df.loc[~df['time'].isna()]
    df = _clean_method_names_amm(df)
    return df


@_memory.cache
def _read_amm_csv(fname, **kwargs):
    df = pd.read_csv(os.path.join(RESULTS_DIR, fname), **kwargs)
    drop_cols_inplace(df, AMM_DROP_COLS)
    return df


def cifar10_amm():
    # df = pd.read_csv(os.path.join(RESULTS_DIR, 'cifar10.csv'))
    # drop_cols_inplace(df, AMM_DROP_COLS)
    df = _read_amm_csv('cifar10.csv')
    return _clean_amm_results_df(df)


def cifar100_amm():
    # df = pd.read_csv(os.path.join(RESULTS_DIR, 'cifar100.csv'))
    # drop_cols_inplace(df, AMM_DROP_COLS)
    df = _read_amm_csv('cifar100.csv')
    return _clean_amm_results_df(df)


@_memory.cache
def caltech_amm(filt='sobel'):
    """filt must be one of {'sobel','dog5x5'}"""
    df = _read_amm_csv('caltech_{}.csv'.format(filt))

    return _clean_amm_results_df(df, timing_dtype='i8')


@_memory.cache
def ucr_amm(k=128, problem='rbf'):
    """k must be one of {64, 128, 256}"""
    df = _read_amm_csv('ucr_k={}_problem={}.csv'.format(k, problem))
    df['origN'] = df['N'].values
    df['N'] = 1000  # timing is for a test set size of 1000
    if problem == 'softmax':
        df['M'] = k  # to get meaningful timing vs acc comparison

    return _clean_amm_results_df(df, sparse_pareto=False)


def main():
    # print(encode_timings())
    # print(lut_timings())
    # print(scan_timings())
    # print(bolt_amm_timings())
    # print(mithral_amm_timings())
    # print(dense_amm_timings())
    # print(osnap_amm_timings())
    # print(sparse_amm_timings())
    # print(cifar10_amm())
    # print(cifar100_amm())
    # print(caltech_amm(filt='sobel'))
    # print(caltech_amm(filt='dog5x5'))
    # print(ucr_amm(k=64))
    # df = ucr_amm(k=128, problem='rbf')

    # df_bolt = df.loc[df['method'] == 'Bolt']
    # print("number of uniq bolt speedups:")
    # print(df_bolt['Speedup'].unique().size)
    # import sys; sys.exit()

    # df = df.loc[df['method'] == 'Bolt']
    # print("bolt dset counts")
    # print(df.groupby('task_id')['task_id'].count())
    # # print("bolt speedup per dset counts")
    # # print(df.groupby('task_id')['Speedup'].unique().count())
    # print("number of uniq speedups:")
    # print(df['Speedup'].unique().size)

    df = cifar10_amm()
    # df = cifar100_amm()
    df = df.loc[df['method'].isin(['Brute Force', 'Mithral', 'SparsePCA'])]
    df = df.sort_values(['method', 'Speedup'], axis=0)
    print(df['method Speedup Accuracy'.split()])


if __name__ == '__main__':
    main()
