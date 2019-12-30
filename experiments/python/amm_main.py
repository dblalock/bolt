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
from . import compress

from . import amm_methods as methods

from joblib import Memory
_memory = Memory('.', verbose=0)


NUM_TRIALS = 5


# @_memory.cache
def _estimator_for_method_id(method_id, **method_hparams):
    return methods.METHOD_TO_ESTIMATOR[method_id](**method_hparams)


def _hparams_for_method(method_id):
    if method_id in methods.SKETCH_METHODS:
        # dvals = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]  # d=1 undef on fd methods
        dvals = [1, 2, 4, 8, 16, 32, 64, 128]
        # dvals = [32] # TODO rm after debug
        # dvals = [16] # TODO rm after debug
        # dvals = [8] # TODO rm after debug
        # dvals = [4] # TODO rm after debug
        # dvals = [3] # TODO rm after debug
        # dvals = [2] # TODO rm after debug
        # dvals = [1] # TODO rm after debug
        if method_id == methods.METHOD_SPARSE_PCA:
            alpha_vals = (.03125, .0625, .125, .25, .5, 1, 2, 4, 8)
            # alpha_vals = (.0625, .125, .25, .5, 1, 2, 4, 8)
            # alpha_vals = (.0625, .125)
            # alpha_vals = [.0625]
            # alpha_vals = (2, 4, 5)
            # alpha_vals = [.1]
            # alpha_vals = [1.]
            # alpha_vals = [10.]
            # alpha_vals = [20.]
            # alpha_vals = [50.]
            return [{'d': d, 'alpha': alpha}
                    for d in dvals for alpha in alpha_vals]
        return [{'d': dval} for dval in dvals]

    if method_id in methods.VQ_METHODS:
        # mvals = [1, 2, 4, 8, 16, 32, 64]
        # mvals = [4, 8, 16, 32, 64]
        # mvals = [1, 2, 4, 8, 16]
        # mvals = [1, 2, 4, 8]
        # mvals = [8, 16] # TODO rm after debug
        # mvals = [8, 16, 64] # TODO rm after debug
        # mvals = [128] # TODO rm after debug
        # mvals = [64] # TODO rim after debug
        # mvals = [32] # TODO rm after debug
        # mvals = [16] # TODO rm after debug
        mvals = [8] # TODO rm after debug
        # mvals = [4] # TODO rm after debug
        # mvals = [1] # TODO rm after debug

        if method_id == methods.METHOD_MITHRAL:
            lut_work_consts = (2, 4, -1)
            # lut_work_consts = [-1] # TODO rm
            params = []
            for m in mvals:
                for const in lut_work_consts:
                    params.append({'ncodebooks': m, 'lut_work_const': const})
            return params

        return [{'ncodebooks': m} for m in mvals]
    if method_id == methods.METHOD_EXACT:
        return [{}]

    raise ValueError(f"Unrecognized method: '{method_id}'")


def _ntrials_for_method(method_id, ntasks):
    # return 1 # TODO rm
    if ntasks > 1:  # no need to avg over trials if avging over multiple tasks
        return 1
    # return NUM_TRIALS if method_id in methods.NONDETERMINISTIC_METHODS else 1
    return NUM_TRIALS if method_id in methods.RANDOM_SKETCHING_METHODS else 1


# ================================================================ metrics

def _compute_compression_metrics(ar):
    # if quantize_to_type is not None:
    #     ar = ar.astype(quantize_to_type)
    # ar -= np.min(ar)
    # ar /= (np.max(ar) / 65535)  # 16 bits
    # ar -= 32768  # center at 0
    # ar = ar.astype(np.int16)

    # elem_sz = ar.dtype.itemsize
    # return {'nbytes_raw': ar.nbytes,
    #         'nbytes_blosc_noshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.NOSHUFFLE)),
    #         'nbytes_blosc_byteshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.SHUFFLE)),
    #         'nbytes_blosc_bitshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.BITSHUFFLE)),
    #         'nbytes_zstd': len(_zstd_compress(ar)),
    #         'nbits_cost': nbits_cost(ar).sum() // 8,
    #         'nbits_cost_zigzag':
    #             nbits_cost(zigzag_encode(ar), signed=False).sum() // 8,
    #         'nbytes_sprintz': compress.sprintz_packed_size(ar)
    #         }

    return {'nbytes_raw': ar.nbytes,
            'nbytes_sprintz': compress.sprintz_packed_size(ar)}


def _compute_metrics(task, Y_hat, compression_metrics=True, **sink):
    Y = task.Y_test
    diffs = Y - Y_hat
    raw_mse = np.mean(diffs * diffs)
    normalized_mse = raw_mse / np.var(Y)
    ynorm = np.linalg.norm(Y) + 1e-20
    yhat_norm = np.linalg.norm(Y_hat) + 1e-20
    r = ((Y / ynorm) * (Y_hat / yhat_norm)).sum()
    metrics = {'raw_mse': raw_mse, 'y_std': Y.std(), 'r': r,
               'normalized_mse': normalized_mse, 'bias': diffs.mean(),
               'y_mean': Y.mean()}
    if compression_metrics:

        # Y_q = compress.quantize(Y, nbits=8)
        # Y_hat_q = compress.quantize(Y_hat, nbits=8)
        # diffs_q = Y_q - Y_hat_q
        # # diffs_q = compress.zigzag_encode(diffs_q).astype(np.uint8)
        # assert Y_q.dtype == np.int8
        # assert diffs_q.dtype == np.int8

        Y_q = compress.quantize(Y, nbits=12)
        Y_hat_q = compress.quantize(Y_hat, nbits=12)
        diffs_q = Y_q - Y_hat_q
        assert Y_q.dtype == np.int16
        assert diffs_q.dtype == np.int16

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

    if task.info:
        if task.info['problem'] == 'classify_linear':
            b = task.info['biases']
            logits_amm = Y_hat + b
            logits_orig = Y + b
            lbls_amm = np.argmax(logits_amm, axis=1).astype(np.int32)
            lbls_orig = np.argmax(logits_orig, axis=1).astype(np.int32)
        elif task.info['problem'] == '1nn':
            lbls_centroids = task.info['lbls_centroids']
            lbls_hat = []
            W = task.W_test
            centroid_norms_sq = (W * W).sum(axis=0)
            sample_norms_sq = (task.X_test * task.X_test).sum(
                axis=1, keepdims=True)
            for prods in [Y_hat, Y]:
                prods = Y_hat
                dists_sq_hat = (-2 * prods) + centroid_norms_sq + sample_norms_sq
                # assert np.min(dists_sq_hat) > -1e-5  # sanity check
                centroid_idx = np.argmin(dists_sq_hat, axis=1)
                lbls_hat.append(lbls_centroids[centroid_idx])
            lbls_amm, lbls_orig = lbls_hat
        lbls = task.info['lbls_test'].astype(np.int32)
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
                            Y_train, **kwargs):
    est = _estimator_for_method_id(method_id, **hparams_dict)
    est.fit(X_train, W_train, Y=Y_train, **kwargs)
    return est


# def _main(tasks, methods=['SVD'], saveas=None, ntasks=None,
def _main(tasks, methods=None, saveas=None, ntasks=None,
          verbose=3, limit_ntasks=2, compression_metrics=False):
    methods = methods.DEFAULT_METHODS if methods is None else methods
    if isinstance(methods, str):
        methods = [methods]
    if limit_ntasks is None or limit_ntasks < 1:
        limit_ntasks = np.inf
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
                if verbose > 3:
                    print("got hparams: ")
                    pprint.pprint(hparams_dict)

                try:
                    est = _fitted_est_for_hparams(
                        method_id, hparams_dict,
                        task.X_train, task.W_train, task.Y_train)
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


# def main_ecg(methods=None, saveas='ecg', limit_nhours=1):
#     tasks = md.load_ecg_tasks(limit_nhours=limit_nhours)
#     return _main(tasks=tasks, methods=methods, saveas=saveas, ntasks=139,
#                  # limit_ntasks=10, compression_metrics=False)
#                  limit_ntasks=5, compression_metrics=True)


def main_caltech(methods=methods.USE_METHODS, saveas='caltech'):
    tasks = md.load_caltech_tasks()
    # tasks = md.load_caltech_tasks(limit_ntrain=100e3, limit_ntest=10e3) # TODO rm after debug
    # tasks = md.load_caltech_tasks(limit_ntrain=-1, limit_ntest=10e3) # TODO rm after debug
    # tasks = md.load_caltech_tasks(limit_ntrain=100e3)
    # tasks = md.load_caltech_tasks(limit_ntrain=500e3)
    # tasks = md.load_caltech_tasks(limit_ntrain=1e6)  # does great
    # tasks = md.load_caltech_tasks(limit_ntrain=15e5)
    # tasks = md.load_caltech_tasks(limit_ntrain=17.5e5) # bad
    # tasks = md.load_caltech_tasks(limit_ntrain=2e6)
    # tasks = md.load_caltech_tasks(limit_ntrain=2.5e6)
    return _main(tasks=tasks, methods=methods, saveas=saveas,
                 # ntasks=510, limit_ntasks=10)
                 # ntasks=510, limit_ntasks=2)
                 ntasks=510, limit_ntasks=3)


def main_ucr(methods=methods.USE_METHODS, saveas='ucr'):
    limit_ntasks = None
    tasks = md.load_ucr_tasks(limit_ntasks=limit_ntasks)
    return _main(tasks=tasks, methods=methods, saveas=saveas,
                 ntasks=76, limit_ntasks=limit_ntasks)


def main_cifar10(methods=methods.USE_METHODS, saveas='cifar10'):
    tasks = md.load_cifar10_tasks()
    return _main(tasks=tasks, methods=methods, saveas=saveas, ntasks=1)


def main_cifar100(methods=methods.USE_METHODS, saveas='cifar100'):
    tasks = md.load_cifar100_tasks()
    return _main(tasks=tasks, methods=methods, saveas=saveas, ntasks=1)


def main_all(methods=methods.USE_METHODS):
    main_cifar10(methods=methods)
    main_cifar100(methods=methods)
    # main_ecg(methods=methods)
    main_caltech(methods=methods)


def main():
    # main_cifar10(methods='SparsePCA')
    # main_cifar10(methods=['OSNAP', 'HashJL'])
    # main_cifar100(methods=['OSNAP', 'HashJL'])
    # main_cifar100(methods='OSNAP')
    # main_cifar10(methods=methods.USE_METHODS)
    # main_cifar100(methods=methods.USE_METHODS)
    main_caltech(methods=methods.USE_METHODS)
    # main_cifar10(methods='Mithral')
    # main_cifar100(methods='Mithral')
    # main_caltech(methods='Mithral')
    # main_caltech(methods='PCA')
    # main_caltech(methods='RandGauss')
    # main_caltech(methods='Hadamard')
    # main_caltech(methods='Rademacher')
    # main_caltech(methods='OrthoGauss')
    # main_caltech(methods='FastJL')
    # main_caltech(methods=['FastJL', 'HashJL', 'OSNAP'])
    # main_caltech(methods='Bolt')
    # main_caltech(methods=['Mithral', 'MithralPQ'])
    # main_cifar10(methods=methods.SLOW_SKETCH_METHODS)
    # main_cifar100(methods=methods.SLOW_SKETCH_METHODS)
    # main_cifar100(methods=['Mithral', 'MithralPQ', 'Bolt', 'Exact', 'PCA', 'FastJL', 'HashJL', 'OSNAP'])
    # main_cifar10(methods=['Mithral', 'MithralPQ', 'Bolt', 'Exact', 'PCA', 'FastJL', 'HashJL', 'OSNAP'])
    # main_cifar10(methods=['MithralPQ', 'Bolt'])
    # main_cifar100(methods=['MithralPQ', 'Bolt'])
    # main_cifar100(methods=['MithralPQ', 'Bolt', 'Exact', 'PCA', 'FastJL', 'HashJL', 'OSNAP'])
    # main_cifar10(methods=['Bolt', 'Exact'])
    # main_cifar10(methods=['MithralPQ', 'Bolt+MultiSplits', 'Bolt', 'Exact'])
    # main_cifar10(methods=['MithralPQ', 'Exact'])
    # main_cifar10(methods='Mithral')
    # main_cifar10(methods='Bolt')
    # main_cifar10(methods=['SparsePCA', 'PCA'])
    # main_cifar100(methods=['SparsePCA', 'PCA'])
    # main_cifar10(methods='MithralPQ')
    # main_cifar100(methods='Mithral')
    # main_cifar100(methods='MithralPQ')
    # main_cifar100(methods=['Mithral', 'MithralPQ', 'Bolt'])
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
    # main_cifar10(methods='Exact')
    # main_ecg(methods='Exact')
    # main_ecg(methods='Bolt')
    # main_ecg(methods='Mithral')
    # main_ecg(methods=['Bolt', 'Exact'])
    # main_ecg(methods=['Bolt', 'Bolt+Perm'])
    # main_caltech(methods=['Bolt+Perm', 'Bolt'])
    # main_caltech(methods=['Exact', 'Bolt'])
    # main_ucr(methods=['Exact', 'Bolt'])
    # main_ucr(methods=['Exact'])
    # main_ucr(methods=['SparsePCA', 'PCA'])

    # imgs = md._load_caltech_train_imgs()
    # imgs = md._load_caltech_test_imgs()


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.2f}".format(f)},
                        linewidth=100)
    main()
