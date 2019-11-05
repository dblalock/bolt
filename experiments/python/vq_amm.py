#!/usr/bin/env python

import numpy as np

from . import vquantizers as vq
from . import amm

KEY_NLOOKUPS = 'nlookups'


class PQMatmul(amm.ApproxMatmul):

    def __init__(self, ncodebooks, ncentroids=None):
        self.ncodebooks = ncodebooks
        self.ncentroids = (self._get_ncentroids() if ncentroids is None
                           else ncentroids)
        self.enc = self._create_encoder(ncodebooks)
        self._reset()

    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    def _get_encoder_kwargs(self):  # to be overriden by subclasses
        return {}

    def _get_ncentroids(self):
        return 256

    def _reset(self):
        self.A_enc = None
        self.luts = None

    def fit(self, A, B, Y=None):
        self.enc.fit(A, B.T)

    def set_A(self, A):
        self.A_enc = self.enc.encode_X(A)

    def set_B(self, B):
        self.luts = self.enc.encode_Q(B.T)

    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.dists_enc(self.A_enc, self.luts)
        # # print("about to compute dists")
        # # d_hat = self.enc.dists_enc(self.A_enc, self.luts)
        # d_hat = self.enc.dists_enc(self.A_enc, self.luts, A, B)
        # d_pad = self.enc._pad_ncols(A) @ self.enc._pad_ncols(B.T).T
        # d = A @ B
        # print("corr(d_hat, d)")
        # print(np.corrcoef(np.vstack([d_hat.ravel(), d.ravel()])))
        # diffs = d - d_hat
        # print("normalized mse of d_hat vs d", np.mean(diffs * diffs) / np.var(d))
        # # diffs = d - d_pad
        # # print("normalized mse of d_pad vs d", np.mean(diffs * diffs) / np.var(d))
        # # import sys; sys.exit()

        # return d_hat

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else A.shape[0] * A.shape[1] * self.ncentroids
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        nlookups = A.shape[0] * B.shape[1] * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

    def get_params(self):
        return {'ncodebooks': self.ncodebooks}


class BoltMatmul(PQMatmul):

    def __init__(self, ncodebooks):
        self.ncodebooks = ncodebooks
        self.ncentroids = 16
        self.enc = self._create_encoder(self.ncodebooks)
        self._reset()

    def _create_encoder(self, ncodebooks):
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            quantize_lut=True,
                            # accumulate_how='mean',
                            # TODO set quantize_lut=True after debug
                            **self._get_encoder_kwargs())


class OPQMatmul(PQMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='OPQ')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        rot_nmuls = A.shape[0] * A.shape[1] * A.shape[1]  # OPQ rotation cost
        metrics[amm.KEY_NMULTIPLIES] += rot_nmuls
        return metrics


class GEHTBoltMatmul_CovTopk(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='deterministic', stats_mat='cov')


class GEHTBoltMatmul_CovSamp(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='importance', stats_mat='cov')


class GEHTBoltMatmul_CorrTopk(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='deterministic', stats_mat='corr')


class GEHTBoltMatmul_CorrSamp(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='importance', stats_mat='corr')


class BoltSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='PQ', encode_algo='splits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class BoltMultiSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(encode_algo='multisplits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class BoltPermMultiSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT', encode_algo='multisplits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQPerm(PQMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQMultiSplits(PQMatmul):

    def __init__(self, ncodebooks, ncentroids=256):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids)

    def _get_encoder_kwargs(self):
        return dict(encode_algo='multisplits')

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQPermMultiSplits(PQMatmul):

    def __init__(self, ncodebooks, ncentroids=256):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids)

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT', encode_algo='multisplits')

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics
