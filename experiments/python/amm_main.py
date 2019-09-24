#!/bin/env/python

import blosc  # pip install blosc
import numpy as np
import pprint
import zstandard as zstd  # pip install zstandard

from . import amm
from . import matmul_datasets as md
from . import pyience as pyn


METHOD_EXACT = 'Exact'
METHOD_SKETCH_SQ_SAMPLE = 'SketchSqSample'
METHOD_SVD = 'SVD'
METHOD_FD_AMM = 'FD-AMM'
METHOD_COOCCUR = 'CooccurSketch'

_METHOD_TO_ESTIMATOR = {
    METHOD_EXACT: amm.ExactMatMul,
    METHOD_SKETCH_SQ_SAMPLE: amm.SketchSqSample,
    METHOD_SVD: amm.SvdSketch,
    METHOD_FD_AMM: amm.FdAmm,
    METHOD_COOCCUR: amm.CooccurSketch,
}
_ALL_METHODS = sorted(list(_METHOD_TO_ESTIMATOR.keys()))
_ALL_METHODS.remove(METHOD_SKETCH_SQ_SAMPLE)  # always terrible results
SKETCH_METHODS = (METHOD_SKETCH_SQ_SAMPLE, METHOD_SVD,
                  METHOD_FD_AMM, METHOD_COOCCUR)
NONDETERMINISTIC_METHODS = (METHOD_SKETCH_SQ_SAMPLE, METHOD_SVD)

NUM_TRIALS = 1  # only for randomized svd, which seems nearly deterministic


def _estimator_for_method_id(method_id, **method_hparams):
    return _METHOD_TO_ESTIMATOR[method_id](**method_hparams)


# ================================================================ metrics

def _zstd_compress(buff, comp=None):
    comp = zstd.ZstdCompressor() if comp is None else comp
    if isinstance(buff, str):
        buff = bytes(buff, encoding='utf8')
    return comp.compress(buff)


def _zstd_decompress(buff, decomp=None):
    decomp = zstd.ZstdDecompressor() if decomp is None else decomp
    return decomp.decompress(decomp)


def _blosc_compress(buff, elem_sz=8, compressor='zstd', shuffle=blosc.SHUFFLE):
    """Thin wrapper around blosc.compress()

    Params:
        compressor: ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
        shuffle: one of [blosc.SHUFFLE, blosc.BITSHUFFLE, blosc.NOSHUFFLE]
        elem_sz: int, size in bytes of each element in buff (eg, 4 for fp32)
    Returns:
        compressed buffer as bytes object
    """
    # decomp with blosc.decompress(compressed_buff)
    return blosc.compress(buff, typesize=elem_sz,
                          cname=compressor, shuffle=shuffle)


def _compute_compression_metrics(ar, quantize_to_type=np.int16):
    if quantize_to_type is not None:
        ar = ar.astype(quantize_to_type)
    elem_sz = ar.dtype.itemsize
    return {'nbytes_orig': ar.nbytes,
            'nbytes_blosc_noshuf': len(_blosc_compress(
                ar, elem_sz=elem_sz, shuffle=blosc.NOSHUFFLE)),
            'nbytes_blosc_byteshuf': len(_blosc_compress(
                ar, elem_sz=elem_sz, shuffle=blosc.SHUFFLE)),
            'nbytes_blosc_bitshuf': len(_blosc_compress(
                ar, elem_sz=elem_sz, shuffle=blosc.BITSHUFFLE)),
            'nbytes_zstd': len(_zstd_compress(ar))}


def _compute_metrics(Y, Y_hat, compression_metrics=True, **sink):
    diffs = Y - Y_hat
    raw_mse = np.mean(diffs * diffs)
    y_var = np.var(Y)
    r_sq = 1 - raw_mse / y_var
    metrics = {'raw_mse': raw_mse, 'y_var': y_var, 'r_sq': r_sq}
    if compression_metrics:
        metrics.update(_compute_compression_metrics(diffs))
    return metrics


# ================================================================ driver funcs

def _eval_amm(task, est, **metrics_kwargs):
    est.fit(A=task.X_train, B=task.W_train, Y=task.Y_train)
    # print("task: ", task.name)
    # print("X_test shape: ", task.X_test.shape)
    # print("W_test shape: ", task.W_test.shape)
    Y_hat = est.predict(task.X_test, task.W_test)
    return _compute_metrics(task.Y_test, Y_hat, **metrics_kwargs)


# SELF: pick up here by writing func to generate params combos based on
# method_id, and then using pyience to expand them
#   -once we have that, can start getting results with other methods

def _hparams_for_method(method_id):
    if method_id in SKETCH_METHODS:
        # dvals = [1, 2, 4, 8, 16, 32, 64, 128]
        dvals = [4, 8, 16, 32, 64, 128]
        # dvals = [32, 64, 128] # TODO rm after debug
        return [{'d': dval} for dval in dvals]
    return [{}]


def _ntrials_for_method(method_id):
    return NUM_TRIALS if method_id in NONDETERMINISTIC_METHODS else 1
    # return 1 # TODO rm


def _get_all_independent_vars():
    independent_vars = set(['task_id', 'method', 'trial'])
    for method_id in _ALL_METHODS:
        hparams = _hparams_for_method(method_id)[0]
        est = _estimator_for_method_id(method_id, **hparams)
        independent_vars = (independent_vars |
                            set(est.get_params().keys()))
    return independent_vars


# def main_ecg(methods=['SketchSqSample'], saveas='ecg_results', limit_nhours=.25):
def main_ecg(methods=None, saveas='ecg_results', limit_nhours=.25):
    methods = _ALL_METHODS if methods is None else methods
    independent_vars = _get_all_independent_vars()

    # for task in load_caltech_tasks():
    for i, task in enumerate(md.load_ecg_tasks(limit_nhours=limit_nhours)):
        print("-------- running task: {} ({}/{})".format(
            task.name, i + 1, 139))
        metrics_for_task = []
        for method_id in methods:
            ntrials = _ntrials_for_method(method_id)
            # for hparams_dict in _hparams_for_method(method_id)[2:]: # TODO rm
            for hparams_dict in _hparams_for_method(method_id):
                print("running method: ", method_id)
                print("got hparams: ")
                pprint.pprint(hparams_dict)
                est = _estimator_for_method_id(method_id, **hparams_dict)
                try:
                    for trial in range(ntrials):
                        metrics = _eval_amm(task, est)
                        metrics['trial'] = trial
                        metrics['method'] = method_id
                        metrics['task_id'] = task.name
                        metrics.update(hparams_dict)
                        print("got metrics: ")
                        pprint.pprint(metrics)
                        metrics_for_task.append(metrics)
                except amm.InvalidParametersException as e:
                    # hparams don't make sense for this task (eg, D < d)
                    print("hparams apparently invalid: {}".format(e))

        if len(metrics_for_task):
            pyn.save_dicts_as_data_frame(
                metrics_for_task, save_dir='results/amm', name='ecg',
                dedup_cols=independent_vars)

        if i > 2:
            return # TODO rm

    # return metrics_for_task

    # for each rec_id in recording_ids
        # for each method_id in method_ids
            # metrics.append(_train_eval_ecg(rec_id, method_id))


def main():
    main_ecg()


if __name__ == '__main__':
    main()
