#!/bin/env/python

import blosc  # pip install blosc
import numpy as np
import zstandard as zstd  # pip install zstandard

from . import amm
from . import matmul_datasets as md
from . import pyience as pyn


_METHOD_TO_ESTIMATOR = {
    'SketchSqSample': amm.SketchSqSample,
    'Svd': amm.SvdSketch,
    'FdAmm': amm.FdAmm,
    'CooccurSketch': amm.CooccurSketch,
}
_ALL_METHODS = sorted(list(_METHOD_TO_ESTIMATOR.keys()))


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
    return {'orig_nbytes': ar.nbytes,
            'blosc_noshuf_nbytes': len(_blosc_compress(
                ar, elem_sz=elem_sz, shuffle=blosc.NOSHUFFLE)),
            'blosc_byteshuf_nbytes': len(_blosc_compress(
                ar, elem_sz=elem_sz, shuffle=blosc.SHUFFLE)),
            'blosc_bitshuf_nbytes': len(_blosc_compress(
                ar, elem_sz=elem_sz, shuffle=blosc.BITSHUFFLE)),
            'zstd_sz': len(_zstd_compress(ar))}  # == noshuffle, except clevel?


def _compute_metrics(Y, Y_hat, compression_metrics=True, **sink):
    diffs = Y - Y_hat
    metrics = {'raw_mse': np.mean(diffs * diffs),
               'y_var': np.var(Y)}
    if compression_metrics:
        metrics.update(_compute_compression_metrics(diffs))
    return metrics


def _eval_amm(task, est, **metrics_kwargs):
    est.fit(task.X_train, task.Y_train, task.W_train)
    Y_hat = est.predict(task.X_hat, task.W_test)
    return _compute_metrics(task.Y_test, Y_hat, **metrics_kwargs)


def train_eval_ecg(rec_id, method_id):
    all_metrics = []
    # cached_load_matmul_tasks(rec_id)
    # B = W matrix (same for train and test)
    # est = get_estimator(method_id)
    # est.set_B(B)
    # for each window:
    #   task = construct the matmultask
    #   metrics = eval_amm(task, est)


def main_ecg():
    all_metrics = []
    # for each rec_id in recording_ids
        # for each method_id in method_ids
            # metrics.append(_train_eval_ecg(rec_id, method_id))



def main():
    pass


if __name__ == '__main__':
    main()
