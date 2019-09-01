#!/usr/bin/env python

import functools
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats.stats import pearsonr
import seaborn as sb
import time

from collections import namedtuple

# import datasets
import files
import product_quantize as pq
import pyience as pyn

from datasets import neighbors as dsets
from utils import kmeans, top_k_idxs

from joblib import Memory
_memory = Memory('.', verbose=0)

np.set_printoptions(precision=3)

SAVE_DIR = '../results'


# ================================================================ Distances

def dists_elemwise_sq(x, q):
    diffs = x - q
    return diffs * diffs


def dists_elemwise_l1(x, q):
    return np.abs(x - q)


def dists_elemwise_dot(x, q):
    return x * q


# ================================================================ Clustering

def load_dataset_object(which_dataset, **load_dataset_kwargs):
    X_train, Q, X_test, true_nn = dsets.load_dataset(
        which_dataset, **load_dataset_kwargs)
    assert Q.shape[-1] == X_train.shape[-1]

    if isinstance(which_dataset, str):
        name = files.basename(which_dataset, noext=True)
    else:
        name = which_dataset.__name__  # assumes which_dataset is a class

    return Dataset(Q, X_train, X_test, true_nn, name)


Dataset = namedtuple('Dataset', [
    'Q', 'X_train', 'X_test', 'true_nn', 'name'])


# ================================================================ Quantizers

# ------------------------------------------------ Product Quantization

def _learn_centroids(X, ncentroids, nsubvects, subvect_len):
    ret = np.empty((ncentroids, nsubvects, subvect_len))
    for i in range(nsubvects):
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        centroids, labels = kmeans(X_in, ncentroids)
        ret[:, i, :] = centroids

    return ret


def _parse_codebook_params(D, code_bits=-1, bits_per_subvect=-1, nsubvects=-1):
    if nsubvects < 0:
        nsubvects = code_bits // bits_per_subvect
    elif code_bits < 1:
        code_bits = bits_per_subvect * nsubvects
    elif bits_per_subvect < 1:
        bits_per_subvect = code_bits // nsubvects

    ncentroids = int(2 ** bits_per_subvect)
    subvect_len = D // nsubvects

    assert code_bits % bits_per_subvect == 0
    if D % subvect_len:
        print("D, nsubvects, subvect_len = ", D, nsubvects, subvect_len)
        assert D % subvect_len == 0  # TODO rm this constraint

    return nsubvects, ncentroids, subvect_len


def _fit_pq_lut(q, centroids, elemwise_dist_func):
    _, nsubvects, subvect_len = centroids.shape
    assert len(q) == nsubvects * subvect_len

    q = q.reshape((1, nsubvects, subvect_len))
    q_dists_ = elemwise_dist_func(centroids, q)
    q_dists_ = np.sum(q_dists_, axis=-1)

    return np.asfortranarray(q_dists_)  # ncentroids, nsubvects, col-major


class PQEncoder(object):

    def __init__(self, dataset, code_bits=-1, bits_per_subvect=-1,
                 nsubvects=-1, elemwise_dist_func=dists_elemwise_sq):
        X = dataset.X_train
        self.elemwise_dist_func = elemwise_dist_func

        tmp = _parse_codebook_params(X.shape[1], code_bits=code_bits,
                                     bits_per_subvect=bits_per_subvect,
                                     nsubvects=nsubvects)
        self.nsubvects, self.ncentroids, self.subvect_len = tmp
        self.code_bits = int(np.log2(self.ncentroids))

        # for fast lookups via indexing into flattened array
        self.offsets = np.arange(self.nsubvects, dtype=np.int) * self.ncentroids

        self.centroids = _learn_centroids(X, self.ncentroids, self.nsubvects,
                                          self.subvect_len)

    def name(self):
        return "PQ_{}x{}b".format(self.nsubvects, self.code_bits)

    def params(self):
        return {'_algo': 'PQ', '_ncodebooks': self.nsubvects,
                '_code_bits': self.code_bits}

    def encode_X(self, X, **sink):
        idxs = pq._encode_X_pq(X, codebooks=self.centroids)
        return idxs + self.offsets  # offsets let us index into raveled dists

    def encode_q(self, q, **sink):
        return None  # we use fit_query() instead, so fail fast

    def dists_true(self, X, q):
        return np.sum(self.elemwise_dist_func(X, q), axis=-1)

    def fit_query(self, q, **sink):
        self.q_dists_ = _fit_pq_lut(q, centroids=self.centroids,
                                    elemwise_dist_func=self.elemwise_dist_func)

    def dists_enc(self, X_enc, q_unused=None):
        # this line has each element of X_enc index into the flattened
        # version of q's distances to the centroids; we had to add
        # offsets to each col of X_enc above for this to work
        centroid_dists = self.q_dists_.T.ravel()[X_enc.ravel()]
        return np.sum(centroid_dists.reshape(X_enc.shape), axis=-1)


def _learn_best_quantization(luts):  # luts can be a bunch of vstacked luts
    best_loss = np.inf
    best_alpha = None
    best_floors = None
    best_scale_by = None
    for alpha in [.001, .002, .005, .01, .02, .05, .1]:
        alpha_pct = int(100 * alpha)

        # compute quantized luts this alpha would yield
        floors = np.percentile(luts, alpha_pct, axis=0)
        luts_offset = np.maximum(0, luts - floors)

        ceil = np.percentile(luts_offset, 100 - alpha_pct)
        scale_by = 255. / ceil
        luts_quantized = np.floor(luts_offset * scale_by).astype(np.int)
        luts_quantized = np.minimum(255, luts_quantized)

        # compute err
        luts_ideal = (luts - luts_offset) * scale_by
        diffs = luts_ideal - luts_quantized
        loss = np.sum(diffs * diffs)

        if loss <= best_loss:
            best_loss = loss
            best_alpha = alpha
            best_floors = floors
            best_scale_by = scale_by

    return best_floors, best_scale_by, best_alpha


class OPQEncoder(PQEncoder):

    def __init__(self, dataset, code_bits=-1, bits_per_subvect=-1,
                 nsubvects=-1, elemwise_dist_func=dists_elemwise_sq,
                 opq_iters=20, quantize_lut=False, algo='OPQ', **opq_kwargs):
        X = dataset.X_train
        self.elemwise_dist_func = elemwise_dist_func
        self.quantize_lut = quantize_lut
        self.opq_iters = opq_iters
        self.algo = algo

        tmp = _parse_codebook_params(X.shape[1], code_bits=code_bits,
                                     bits_per_subvect=bits_per_subvect,
                                     nsubvects=nsubvects)
        self.nsubvects, self.ncentroids, self.subvect_len = tmp
        self.code_bits = int(np.log2(self.ncentroids))

        # for fast lookups via indexing into flattened array
        self.offsets = np.arange(self.nsubvects, dtype=np.int) * self.ncentroids

        if self.algo == 'Bolt':
            # Note: we always pass in 0 iters in the reported experiments,
            # so it never rotates anything
            self.centroids, _, self.rotations = pq.learn_bopq(
                X, ncodebooks=nsubvects, codebook_bits=bits_per_subvect,
                niters=opq_iters, **opq_kwargs)
        elif self.algo == 'OPQ':
            self.centroids, _, self.R = pq.learn_opq(
                X, ncodebooks=nsubvects, codebook_bits=bits_per_subvect,
                niters=opq_iters, **opq_kwargs)
        else:
            raise ValueError("argument algo must be one of {OPQ, Bolt}")

        # learn appropriate offsets and shared scale factor for quantization
        self.lut_offsets = np.zeros(self.nsubvects)
        self.order_idxs = np.arange(self.nsubvects, dtype=np.int)

        if self.quantize_lut:  # TODO put this logic in separate function
            print("learning quantization...")

            num_rows = min(10*1000, len(X) // 2)
            _, queries = dsets.extract_random_rows(
                X[num_rows:], how_many=1000, remove_from_X=False)
            X = X[:num_rows]  # limit to first 10k rows of X

            # compute luts for all the queries
            luts = [self._fit_query(q, quantize=False) for q in queries]
            luts = np.vstack(luts)
            assert luts.shape == (self.ncentroids * len(queries), self.nsubvects)

            self.lut_offsets, self.scale_by, _ = _learn_best_quantization(luts)

    def name(self):
        return "{}_{}x{}b_iters={}_quantize={}".format(
            self.algo, self.nsubvects, self.code_bits, self.opq_iters,
            int(self.quantize_lut))

    def params(self):
        return {'_algo': self.algo, '_ncodebooks': self.nsubvects,
                '_code_bits': self.code_bits, 'opq_iters': self.opq_iters,
                '_quantize': self.quantize_lut}

    def _fit_query(self, q, quantize=False):
        if self.algo == 'OPQ':
            qR = pq.opq_rotate(q, self.R).ravel()
        elif self.algo == 'Bolt':
            qR = pq.bopq_rotate(q, self.rotations).ravel()
        lut = _fit_pq_lut(qR, centroids=self.centroids,
                          elemwise_dist_func=self.elemwise_dist_func)

        if quantize:
            if False:  # roughly laplace distro, reaching all the way to 0
                ax = sb.distplot(lut.ravel(), hist=False, rug=True)
                ax.set_xlabel('Query dist to centroids (lut dist histogram)')
                ax.set_ylabel('Fraction of queries')
                plt.show()

            lut = np.maximum(0, lut - self.lut_offsets)
            lut = np.floor(lut * self.scale_by).astype(np.int)
            return np.minimum(lut, 255)

        return lut

    def encode_X(self, X, **sink):
        if self.algo == 'OPQ':
            X = pq.opq_rotate(X, self.R)
        elif self.algo == 'Bolt':
            X = pq.bopq_rotate(X, self.rotations)

        idxs = pq._encode_X_pq(X, codebooks=self.centroids)

        return idxs + self.offsets  # offsets let us index into raveled dists

    def fit_query(self, q, quantize=True, **sink):
        quantize = quantize and self.quantize_lut
        self.q_dists_ = self._fit_query(q, quantize=quantize)

        if quantize:
            # print "min, max lut values: {}, {}".format(np.min(self.q_dists_),
            #     np.max(self.q_dists_))
            assert np.min(self.q_dists_) >= 0
            assert np.max(self.q_dists_) <= 255

        if False:
            _, axes = plt.subplots(3, figsize=(9, 11))
            sb.violinplot(data=self.q_dists_, inner="box", cut=0, ax=axes[0])
            axes[0].set_xlabel('Codebook')
            axes[0].set_ylabel('Distance to query')
            axes[0].set_ylim([0, np.max(self.q_dists_)])

            sb.heatmap(data=self.q_dists_, ax=axes[1], cbar=False, vmin=0)
            axes[1].set_xlabel('Codebook')
            axes[1].set_ylabel('Centroid')

            sb.distplot(self.q_dists_.ravel(), hist=False, rug=True, vertical=False, ax=axes[2])
            axes[2].set_xlabel('Centroid dist to query')
            axes[2].set_ylabel('Fraction of centroids')
            axes[2].set_xlim([0, np.max(self.q_dists_) + .5])

            # plot where the mean is
            mean_dist = np.mean(self.q_dists_)
            ylim = axes[2].get_ylim()
            axes[2].plot([mean_dist, mean_dist], ylim, 'r--')
            axes[2].set_ylim(ylim)

            plt.show()


# ================================================================ Main

def eval_encoder(dataset, encoder, dist_func_true=None, dist_func_enc=None,
                 eval_dists=True, verbosity=1, plot=False, smaller_better=True):

    X = dataset.X_test
    queries = dataset.Q
    true_nn = dataset.true_nn

    if true_nn is not None:
        print("eval encoder(): got true_nn with shape: ", true_nn.shape)

    queries = queries[:1000]  # TODO rm for tables; fine for plots
    print("queries.shape", queries.shape)

    need_true_dists = eval_dists or plot or true_nn is None

    if len(queries.shape) == 1:
        queries = [queries]

    if dist_func_true is None:
        dist_func_true = encoder.dists_true
    if dist_func_enc is None:
        dist_func_enc = encoder.dists_enc

    t0 = time.time()

    # performance metrics
    RECALL_Rs = [1, 5, 10, 50, 100, 500, 1000]
    recall_counts = np.zeros(len(RECALL_Rs))
    fracs_below_max = []
    if eval_dists:
        all_corrs = []
        all_rel_errs = []
        all_errs = []
        total_dist = 0.

    if need_true_dists:
        X = X[:10000]  # limit to 10k points because otherwise it takes forever
        queries = queries[:256, :]

    print("encoding X...")
    X_enc = encoder.encode_X(X)
    print("trying queries...")
    for i, q in enumerate(queries):

        if i % 100 == 0:
            print("trying query {}...".format(i))

        q_enc = encoder.encode_q(q)
        encoder.fit_query(q)
        if need_true_dists:
            all_true_dists = dist_func_true(X, q)

        all_enc_dists = dist_func_enc(X_enc, q_enc)

        # ------------------------ begin analysis / reporting code

        # find true knn
        if need_true_dists:
            knn_idxs = top_k_idxs(all_true_dists, 10, smaller_better=smaller_better)
        else:
            knn_idxs = true_nn[i, :10]

        # compute fraction of points with enc dists as close as 10th nn
        knn_enc_dists = all_enc_dists[knn_idxs]
        if smaller_better:
            max_enc_dist = np.max(knn_enc_dists)
            num_below_max = np.sum(all_enc_dists <= max_enc_dist)
        else:
            max_enc_dist = np.min(knn_enc_dists)
            num_below_max = np.sum(all_enc_dists >= max_enc_dist)

        frac_below_max = float(num_below_max) / len(all_enc_dists)
        fracs_below_max.append(frac_below_max)

        # compute recall@R stats
        top_1000 = top_k_idxs(all_enc_dists, 1000, smaller_better=smaller_better)
        nn_idx = knn_idxs[0]
        for i, r in enumerate(RECALL_Rs):
            recall_counts[i] += nn_idx in top_1000[:r]

        # compute distortion in distances, quantified by corr and rel err
        if eval_dists:
            total_dist += np.sum(all_true_dists)
            corr, _ = pearsonr(all_enc_dists, all_true_dists)
            all_corrs.append(corr)
            rel_errs = (all_enc_dists - all_true_dists) / all_true_dists
            all_rel_errs.append(rel_errs)
            all_errs.append(all_enc_dists - all_true_dists)
            assert not np.any(np.isinf(all_enc_dists))
            assert not np.any(np.isnan(all_enc_dists))
            assert not np.any(np.isinf(all_true_dists))
            assert not np.any(np.isnan(all_true_dists))

        if plot and i < 3:  # at most 3 plots
            num_nn = min(10000, len(all_true_dists) - 1)
            xlim = [0, np.partition(all_true_dists, num_nn)[num_nn]]
            ylim = [0, np.partition(all_enc_dists, num_nn)[num_nn]]

            grid = sb.jointplot(x=all_true_dists, y=all_enc_dists,
                                xlim=xlim, ylim=ylim, joint_kws=dict(s=10))

            # hack to bully the sb JointGrid into plotting a vert line
            cutoff = all_true_dists[knn_idxs[-1]]
            grid.x = [cutoff, cutoff]
            grid.y = ylim
            grid.plot_joint(plt.plot, color='r', linestyle='--')

            # also make it plot cutoff in terms of quantized dist
            grid.x = xlim
            grid.y = [max_enc_dist, max_enc_dist]
            grid.plot_joint(plt.plot, color='k', linestyle='--')

    if plot:
        plt.show()

    t = time.time() - t0

    # log a lot of performance metrics / experimental params
    detailed_stats = []  # list of dicts
    stats = {}
    stats['X_rows'] = X.shape[0]
    stats['X_cols'] = X.shape[1]
    stats['nqueries'] = len(queries)
    stats['eval_time_secs'] = t
    stats['fracs_below_max_mean'] = np.mean(fracs_below_max)
    stats['fracs_below_max_std'] = np.std(fracs_below_max)
    stats['fracs_below_max_50th'] = np.median(fracs_below_max)
    stats['fracs_below_max_90th'] = np.percentile(fracs_below_max, q=90)
    for i, r in enumerate(RECALL_Rs):
        key = 'recall@{}'.format(r)
        val = float(recall_counts[i]) / len(queries)
        stats[key] = val
    if eval_dists:
        corrs = np.hstack(all_corrs)
        rel_errs = np.hstack(all_rel_errs)
        rel_errs = rel_errs[~(np.isnan(rel_errs) + np.isinf(rel_errs))]
        errs = np.hstack(all_errs)

        stats['corr_mean'] = np.mean(all_corrs)
        stats['corr_std'] = np.std(all_corrs)
        stats['mse_mean'] = np.mean(errs * errs)
        stats['mse_std'] = np.std(errs * errs)
        stats['rel_err_mean'] = np.mean(rel_errs)
        stats['rel_err_std'] = np.std(rel_errs)
        stats['rel_err_sq_mean'] = np.mean(rel_errs * rel_errs)
        stats['rel_err_sq_std'] = np.std(rel_errs * rel_errs)

        # sample some relative errs cuz all we need them for is plotting
        # confidence intervals
        np.random.shuffle(rel_errs)
        np.random.shuffle(errs)
        detailed_stats = [{'corr': all_corrs[i], 'rel_err': rel_errs[i],
                           'err': errs[i]} for i in range(len(corrs))]

        for d in detailed_stats:
            d.update(encoder_params(encoder))

    if verbosity > 0:
        print("------------------------ {}".format(name_for_encoder(encoder)))
        keys = sorted(stats.keys())
        lines = ["{}: {}".format(k, stats[k]) for k in keys if isinstance(stats[k], str)]
        lines += ["{}: {:.4g}".format(k, stats[k]) for k in keys if not isinstance(stats[k], str)]
        print("\n".join(lines))

    stats.update(encoder_params(encoder))

    return stats, detailed_stats  # detailed_stats empty unless `eval_dists`


def name_for_encoder(encoder):
    try:
        return encoder.name()
    except AttributeError:
        return str(type(encoder))


def encoder_params(encoder):
    try:
        return encoder.params()
    except AttributeError:
        return {'algo': name_for_encoder(encoder)}


# @_memory.cache
def _experiment_one_dataset(which_dataset, eval_dists=False, dotprods=False,
                            save_dir=None):
    SAVE_DIR = save_dir if save_dir else '../results/acc/'

    elemwise_dist_func = dists_elemwise_dot if dotprods else dists_elemwise_sq
    smaller_better = not dotprods

    N, D = -1, -1

    num_queries = -1  # no effect for "real" datasets
    if isinstance(which_dataset, str):
        print("WARNING: sampling queries from data file")
        num_queries = 128  # if just loading one file, need to sample queries

    norm_len = False  # set to true for cosine similarity
    norm_mean = True

    max_ncodebooks = 64  # 32B bolt has 64 codebooks

    dataset_func = functools.partial(load_dataset_object, N=N, D=D,
                                     num_queries=num_queries,
                                     norm_len=norm_len, norm_mean=norm_mean,
                                     D_multiple_of=max_ncodebooks)

    dataset = dataset_func(which_dataset)
    print("=== Using Dataset: {} ({}x{})".format(dataset.name, N, D))

    dicts = []
    detailed_dicts = []
    nbytes_list = [8, 16, 32]
    # max_opq_iters = 5 # TODO uncomment below
    max_opq_iters = 20

    # ------------------------------------------------ Bolt
    # Note: we considered having learned rotations like OPQ but constrained
    # to be block diagonal; this is why you'll see mentions of rotations
    # in some of the Bolt code. However, it ended up not helping at all
    # and also slows down Bolt considerably. All the reported results are
    # without any rotation.

    # rotation_sizes = [8, 16, 32]
    rotation_sizes = [32]
    # rotation_sizes = [16]
    for nbytes in nbytes_list:
        for opq_iters in [0]:  # 0 opq iters -> no rotations
            rot_sizes = rotation_sizes if opq_iters > 0 else [16]
            for rot_sz in rot_sizes:
                nsubvects = nbytes * 2
                encoder = OPQEncoder(dataset, nsubvects=nsubvects,
                                     bits_per_subvect=4,
                                     opq_iters=opq_iters,
                                     R_sz=rot_sz,
                                     elemwise_dist_func=elemwise_dist_func,
                                     algo='Bolt', quantize_lut=True)
                stats, detailed_stats = eval_encoder(
                    dataset, encoder, eval_dists=eval_dists,
                    smaller_better=smaller_better)
                stats['rot_sz'] = rot_sz
                for d in detailed_dicts:
                    d['rot_sz'] = rot_sz
                dicts.append(stats)
                detailed_dicts += detailed_stats

    # ------------------------------------------------ PQ
    # for codebook_bits in [4, 8]:
    for codebook_bits in [8]:
        for nbytes in nbytes_list:
            nsubvects = nbytes * (8 // codebook_bits)
            encoder = PQEncoder(dataset, nsubvects=nsubvects,
                                bits_per_subvect=codebook_bits,
                                elemwise_dist_func=elemwise_dist_func)
            stats, detailed_stats = eval_encoder(
                dataset, encoder, eval_dists=eval_dists,
                smaller_better=smaller_better)
            dicts.append(stats)
            detailed_dicts += detailed_stats

    # ------------------------------------------------ OPQ
    init = 'identity'
    opq_iters = max_opq_iters
    for codebook_bits in [8]:
        for nbytes in nbytes_list:
            nsubvects = nbytes * (8 // codebook_bits)
            encoder = OPQEncoder(dataset, nsubvects=nsubvects,
                                 bits_per_subvect=codebook_bits,
                                 opq_iters=opq_iters, init=init,
                                 elemwise_dist_func=elemwise_dist_func)
            stats, detailed_stats = eval_encoder(
                dataset, encoder, eval_dists=eval_dists,
                smaller_better=smaller_better)
            dicts.append(stats)
            detailed_dicts += detailed_stats

    for d in dicts:
        d['dataset'] = dataset.name
        d['norm_mean'] = norm_mean
    for d in detailed_dicts:
        d['dataset'] = dataset.name
        d['norm_mean'] = norm_mean

    savedir = os.path.join(SAVE_DIR, dataset.name)

    pyn.save_dicts_as_data_frame(dicts, savedir, name='summary')
    # also just save versions with timestamps to recover from clobbering
    pyn.save_dicts_as_data_frame(dicts, savedir, name='summary',
                                 timestamp=True)
    if eval_dists:
        pyn.save_dicts_as_data_frame(detailed_dicts, savedir, name='all_results')
        pyn.save_dicts_as_data_frame(detailed_dicts, savedir, name='all_results',
                                     timestamp=True)

    return dicts, detailed_dicts


def experiment(eval_dists=False, dotprods=False):

    # which_datasets = [dsets.Mnist]
    which_datasets = [dsets.Mnist, dsets.Sift1M,
                      dsets.LabelMe, dsets.Convnet1M]
    # which_datasets = [dsets.Glove]
    # which_datasets = [dsets.Deep1M, dsets.Gist]

    if eval_dists:
        save_dir = '../results/acc_dotprods/' if dotprods else '../results/acc_l2'
    else:
        save_dir = '../results/recall_at_r/'

    for which_dataset in which_datasets:
        _dicts, _details = _experiment_one_dataset(
            which_dataset, eval_dists=eval_dists, dotprods=dotprods,
            save_dir=save_dir)


def main():
    import doctest
    doctest.testmod()
    np.set_printoptions(precision=3)

    opts = pyn.parse_cmd_line()
    opts.setdefault('eval_l2_dists', False)
    opts.setdefault('eval_dotprods', False)
    opts.setdefault('eval_recall@R', False)

    if opts['eval_l2_dists']:
        print(">>>>>>>> evaluating l2 dists")
        experiment(eval_dists=True, dotprods=False)
    if opts['eval_dotprods']:
        print(">>>>>>>> evaluating dot prods")
        experiment(eval_dists=True, dotprods=True)
    if opts['eval_recall@R']:
        print(">>>>>>>> evaluating recall@R")
        experiment(eval_dists=False, dotprods=False)
    return


if __name__ == '__main__':
    main()
