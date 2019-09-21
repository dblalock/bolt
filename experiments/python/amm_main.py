#!/bin/env/python

import blosc  # pip install blosc
import numpy as np
import pprint
import zstandard as zstd  # pip install zstandard

from . import amm
from . import matmul_datasets as md
from . import pyience as pyn


_METHOD_TO_ESTIMATOR = {
    'Exact': amm.ExactMatMul,
    'SketchSqSample': amm.SketchSqSample,
    'Svd': amm.SvdSketch,
    'FdAmm': amm.FdAmm,
    'CooccurSketch': amm.CooccurSketch,
}
_ALL_METHODS = sorted(list(_METHOD_TO_ESTIMATOR.keys()))


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
    Y_hat = est.predict(task.X_test, task.W_test)
    return _compute_metrics(task.Y_test, Y_hat, **metrics_kwargs)


# def train_eval_ecg(rec_id, method_id):
#     all_metrics = []
#     # cached_load_matmul_tasks(rec_id)
#     # B = W matrix (same for train and test)
#     # est = get_estimator(method_id)
#     # est.set_B(B)
#     # for each window:
#     #   task = construct the matmultask
#     #   metrics = eval_amm(task, est)


def main_ecg(methods=['Exact'], saveas='ecg_results'):
    methods = _ALL_METHODS if methods is None else methods

    base_independent_vars = set(['task_id', 'method'])
    for method_id in methods:
        est = _estimator_for_method_id(method_id)
        independent_vars = (base_independent_vars |
                            set(est.get_params().keys()))

    # print("preproc_kwargs: ", preproc_kwargs)
    # independent_vars += list(est.get_params().keys())

    all_metrics = []
    # for task in load_caltech_tasks():
    for i, task in enumerate(md.load_ecg_tasks()):
        print("-------- running task: {} ({}/{})".format(task.name, i + 1, 139)
        for method_id in methods:
            print("running method: ", method_id)
            # TODO sweep hparams somehow; prolly some func to generate set of
            # hparams for a given method id
            est = _estimator_for_method_id(method_id)
            metrics = _eval_amm(task, est)
            metrics['method'] = method_id
            metrics['task_id'] = task.name
            print("got metrics: ")
            pprint.pprint(metrics)
            all_metrics.append(metrics)

    pyn.save_dicts_as_data_frame(
        all_metrics, save_dir='results/amm', name='ecg',
        dedup_cols=independent_vars)
    return all_metrics

    # for each rec_id in recording_ids
        # for each method_id in method_ids
            # metrics.append(_train_eval_ecg(rec_id, method_id))


def main():
    main_ecg()


if __name__ == '__main__':
    main()
