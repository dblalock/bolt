#!/bin/env/python

import blosc  # pip install blosc
# import functools
import numpy as np
import pprint
import time
import zstandard as zstd  # pip install zstandard

from . import amm
from . import matmul_datasets as md
from . import pyience as pyn
# from . import vq_amm

from . import amm_methods as methods

from joblib import Memory
_memory = Memory('.', verbose=0)


NUM_TRIALS = 1  # only for randomized svd, which seems nearly deterministic


def _estimator_for_method_id(method_id, **method_hparams):
    return methods.METHOD_TO_ESTIMATOR[method_id](**method_hparams)


def _hparams_for_method(method_id):
    if method_id in methods.SKETCH_METHODS:
        # dvals = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]  # d=1 undef on fd methods
        dvals = [2, 4, 8, 16, 32, 64, 128]
        # dvals = [32] # TODO rm after debug
        return [{'d': dval} for dval in dvals]
    if method_id in methods.VQ_METHODS:
        # mvals = [1, 2, 4, 8, 16, 32, 64]
        # mvals = [4, 8, 16, 32, 64]
        # mvals = [1, 2, 4, 8, 16]
        # mvals = [1, 2, 4, 8]
        # mvals = [8, 16] # TODO rm after debug
        # mvals = [8, 16, 64] # TODO rm after debug
        # mvals = [16] # TODO rm after debug
        # mvals = [8] # TODO rm after debug
        mvals = [4] # TODO rm after debug
        # mvals = [1] # TODO rm after debug
        return [{'ncodebooks': m} for m in mvals]
    return [{}]


def _ntrials_for_method(method_id, ntasks):
    # return 1 # TODO rm
    if ntasks > 1:  # no need to avg over trials if avging over multiple tasks
        return 1
    return NUM_TRIALS if method_id in methods.NONDETERMINISTIC_METHODS else 1


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


# def _compute_compression_metrics(ar, quantize_to_type=np.uint16):
def _compute_compression_metrics(ar):
    # if quantize_to_type is not None:
    #     ar = ar.astype(quantize_to_type)
    # ar -= np.min(ar)
    # ar /= (np.max(ar) / 65535)  # 16 bits
    # ar -= 32768  # center at 0
    # ar = ar.astype(np.int16)

    elem_sz = ar.dtype.itemsize
    return {'nbytes_raw': ar.nbytes,
            # 'nbytes_blosc_noshuf': len(_blosc_compress(
            #     ar, elem_sz=elem_sz, shuffle=blosc.NOSHUFFLE)),
            'nbytes_blosc_byteshuf': len(_blosc_compress(
                ar, elem_sz=elem_sz, shuffle=blosc.SHUFFLE)),
            'nbytes_blosc_bitshuf': len(_blosc_compress(
                ar, elem_sz=elem_sz, shuffle=blosc.BITSHUFFLE)),
            'nbytes_zstd': len(_zstd_compress(ar))}


def _compute_metrics(task, Y_hat, compression_metrics=True, **sink):
    Y = task.Y_test
    diffs = Y - Y_hat
    raw_mse = np.mean(diffs * diffs)
    y_var = np.var(Y)
    r_sq = 1 - raw_mse / y_var
    metrics = {'raw_mse': raw_mse, 'y_var': y_var, 'r_sq': r_sq}
    if compression_metrics:

        def quantize(X, minval, maxval, nbits=16):
            upper = (1 << nbits) - 1
            dtype_min = 1 << (nbits - 1)

            X = np.maximum(0, X - minval)
            X = np.minimum(upper, (X / maxval) * upper)
            X -= dtype_min  # center at 0

            dtype = {16: np.int16, 12: np.int16, 8: np.int8}[nbits]
            return X.astype(dtype)

        minval = np.min(Y)
        maxval = np.max(Y)
        Y_q = quantize(Y, minval, maxval, nbits=8)
        Y_hat_q = quantize(Y_hat, minval, maxval, nbits=8)
        diffs_q = Y_q - Y_hat_q
        assert Y_q.dtype == np.int8
        assert diffs_q.dtype == np.int8

        # Y_q = quantize_i16(Y)

        # # quantize to 16 bits
        # Y = Y - np.min(Y)
        # Y /= (np.max(Y) / 65535)  # 16 bits
        # Y -= 32768  # center at 0
        # Y = Y.astype(np.int16)
        # diffs =

        metrics_raw = _compute_compression_metrics(Y_q)
        metrics.update({k + '_orig': v for k, v in metrics_raw.items()})
        metrics_raw = _compute_compression_metrics(diffs_q)
        metrics.update({k + '_diffs': v for k, v in metrics_raw.items()})

    # eval softmax accuracy TODO better criterion for when to try this
    if task.info:
        b = task.info['biases']
        logits_amm = Y_hat + b
        logits_orig = Y + b
        lbls = task.info['lbls_test'].astype(np.int32)
        lbls_amm = np.argmax(logits_amm, axis=1).astype(np.int32)
        lbls_orig = np.argmax(logits_orig, axis=1).astype(np.int32)
        metrics['acc_amm'] = np.mean(lbls_amm == lbls)
        metrics['acc_orig'] = np.mean(lbls_orig == lbls)

    return metrics


# ================================================================ driver funcs

def _eval_amm(task, est, fixedB=True, **metrics_kwargs):
    if fixedB:
        est.set_B(task.W_test)
    # print("task: ", task.name)
    # print("X_test shape: ", task.X_test.shape)
    # print("W_test shape: ", task.W_test.shape)
    t = time.perf_counter()
    Y_hat = est.predict(task.X_test, task.W_test)
    duration_secs = time.perf_counter() - t

    metrics = _compute_metrics(task, Y_hat, **metrics_kwargs)
    metrics['secs'] = duration_secs
    # metrics['nmultiplies'] = est.get_nmuls(task.X_test, task.W_test)
    metrics.update(est.get_speed_metrics(
        task.X_test, task.W_test, fixedB=fixedB))
    return metrics


def _get_all_independent_vars():
    independent_vars = set(['task_id', 'method', 'trial'])
    for method_id in methods.ALL_METHODS:
        hparams = _hparams_for_method(method_id)[0]
        est = _estimator_for_method_id(method_id, **hparams)
        independent_vars = (independent_vars |
                            set(est.get_params().keys()))
    return independent_vars


# @functools.lru_cache(maxsize=None)
# @_memory.cache
def _fitted_est_for_hparams(method_id, hparams_dict, X_train, W_train,
                            **kwargs):
    est = _estimator_for_method_id(method_id, **hparams_dict)
    est.fit(X_train, W_train, **kwargs)
    return est


# def _main(tasks, methods=['SVD'], saveas=None, ntasks=None,
def _main(tasks, methods=None, saveas=None, ntasks=None,
          verbose=2, limit_ntasks=2, compression_metrics=False):
    methods = methods.DEFAULT_METHODS if methods is None else methods
    if isinstance(methods, str):
        methods = [methods]
    independent_vars = _get_all_independent_vars()

    # for task in load_caltech_tasks():
    for i, task in enumerate(tasks):
        if verbose > 0:
            print("-------- running task: {} ({}/{})".format(
                task.name, i + 1, ntasks))
        task.validate_shapes()  # fail fast if task is ill-formed
        for method_id in methods:
            if verbose > 1:
                print("running method: ", method_id)
            ntrials = _ntrials_for_method(method_id=method_id, ntasks=ntasks)
            # for hparams_dict in _hparams_for_method(method_id)[2:]: # TODO rm
            metrics_dicts = []
            for hparams_dict in _hparams_for_method(method_id):
                if verbose > 2:
                    print("got hparams: ")
                    pprint.pprint(hparams_dict)

                est = _fitted_est_for_hparams(
                    method_id, hparams_dict, task.X_train, task.W_train)
                try:
                    for trial in range(ntrials):
                        metrics = _eval_amm(
                            task, est, compression_metrics=compression_metrics)
                        metrics['N'] = task.X_test.shape[0]
                        metrics['D'] = task.X_test.shape[1]
                        metrics['M'] = task.W_test.shape[1]
                        metrics['trial'] = trial
                        metrics['method'] = method_id
                        metrics['task_id'] = task.name
                        # metrics.update(hparams_dict)
                        metrics.update(est.get_params())
                        print("got metrics: ")
                        pprint.pprint(metrics)
                        metrics_dicts.append(metrics)
                except amm.InvalidParametersException as e:
                    # hparams don't make sense for this task (eg, D < d)
                    if verbose > 2:
                        print("hparams apparently invalid: {}".format(e))

            if len(metrics_dicts):
                pyn.save_dicts_as_data_frame(
                    metrics_dicts, save_dir='results/amm', name=saveas,
                    dedup_cols=independent_vars)

        if i + 1 >= limit_ntasks:
            return


def main_ecg(methods=None, saveas='ecg', limit_nhours=1):
    tasks = md.load_ecg_tasks(limit_nhours=limit_nhours)
    return _main(tasks=tasks, methods=methods, saveas=saveas, ntasks=139,
                 # limit_ntasks=10, compression_metrics=True)
                 limit_ntasks=10, compression_metrics=False)


def main_caltech(methods=None, saveas='caltech'):
    tasks = md.load_caltech_tasks()
    return _main(tasks=tasks, methods=methods, saveas=saveas,
                 ntasks=510, limit_ntasks=10)


def main_cifar10(methods=None, saveas='cifar10'):
    tasks = md.load_cifar10_tasks()
    return _main(tasks=tasks, methods=methods, saveas=saveas, ntasks=1)


def main_cifar100(methods=None, saveas='cifar100'):
    tasks = md.load_cifar100_tasks()
    return _main(tasks=tasks, methods=methods, saveas=saveas, ntasks=1)


def main_all(methods=None):
    main_cifar10(methods=methods)
    main_cifar100(methods=methods)
    main_ecg(methods=methods)
    main_caltech(methods=methods)


def main():
    # main_cifar10(methods=['MithralPQ', 'Bolt', 'Exact', 'PCA', 'FastJL', 'HashJL', 'OSNAP'])
    # main_cifar10(methods=['MithralPQ', 'Bolt'])
    # main_cifar100(methods=['MithralPQ', 'Bolt'])
    # main_cifar100(methods=['MithralPQ', 'Bolt', 'Exact', 'PCA', 'FastJL', 'HashJL', 'OSNAP'])
    # main_cifar10(methods=['Bolt', 'Exact'])
    # main_cifar10(methods=['MithralPQ', 'Bolt+MultiSplits', 'Bolt', 'Exact'])
    # main_cifar10(methods=['MithralPQ', 'Exact'])
    main_cifar10(methods='Mithral')
    # main_cifar100(methods='Mithral')
    # main_cifar10(methods='MithralPQ')
    # main_cifar100(methods='MithralPQ')
    # main_cifar10(methods=['Mithral', 'MithralPQ', 'Bolt'])
    # main_cifar10(methods=['PCA', 'Exact'])
    # main_cifar10(methods=['PCA', 'FastJL', 'HashJL', 'OSNAP', 'Exact'])
    # main_cifar100(methods=['PCA', 'Exact'])
    # main_cifar10(methods=['Bolt+MultiSplits', 'Bolt', 'Exact'])
    # main_cifar10(methods=['Bolt+MultiSplits', 'Bolt'])
    # main_cifar10(methods='Bolt+MultiSplits')
    # main_ecg(methods=['Bolt', 'Exact'])
    # main_ecg(methods='Exact')
    # main_cifar10(methods=['OPQ', 'Exact'])
    # main_cifar100(methods=['PQ', 'Exact'])
    # main_cifar10(methods=VQ_METHODS)
    # main_cifar100(methods=VQ_METHODS)
    # main_cifar10(methods=BOLT_METHODS)
    # main_cifar10(methods=['Bolt_CovTopk'])
    # main_cifar10(methods=['GEHTBoltMatmul_CovSamp'])
    # main_cifar10(methods=['GEHTBoltMatmul_CorrSamp'])
    # main_cifar10(methods=['GEHTBoltMatmul_CorrTopk'])
    # main_cifar100(methods=['GEHTBoltMatmul_CovTopk'])
    # main_cifar100(methods=['Bolt', 'Bolt+Perm'])
    # main_cifar10(methods=['Bolt', 'Exact'])
    # main_cifar10(methods=['Bolt'])
    # main_cifar10(methods=['BoltSplits', 'Bolt', 'PQSplits', 'PQ'])
    # main_cifar10(methods=['PQ+Ours', 'PQ'])
    # main_cifar10(methods=['PQ+MultiSplits', 'PQ+Perm+MultiSplits', 'PQ+Perm', 'PQ', 'Bolt'])  # noqa
    # main_cifar10(methods=['Bolt+MultiSplits', 'Bolt+Perm+MultiSplits', 'Bolt'])  # noqa
    # main_cifar100(methods=['Bolt+MultiSplits', 'Bolt+Perm+MultiSplits', 'Bolt'])  # noqa
    # main_cifar10(methods=['BoltSplits'])
    # main_cifar100(methods=['BoltSplits'])
    # main_cifar100(methods=['Bolt', 'BoltSplits'])
    # main_cifar10()
    # main_cifar100()
    # main_ecg()
    # main_caltech()
    # main_ecg(methods=['Bolt+Perm', 'Bolt+CorrPerm', 'Bolt'])
    # main_ecg(methods=['PQ', 'Bolt', 'Exact'])
    # main_ecg(methods=['Bolt', 'Exact'])
    # main_ecg(methods=['Bolt', 'PQ', 'Exact'])
    # main_caltech(methods=['Bolt', 'PQ', 'Exact'])
    # main_ecg(methods='Bolt')
    # main_ecg(methods=['Bolt', 'Bolt+Perm'])
    # main_caltech(methods=['Bolt+Perm', 'Bolt'])

    # imgs = md._load_caltech_train_imgs()
    # imgs = md._load_caltech_test_imgs()


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.2f}".format(f)},
                        linewidth=100)
    main()
