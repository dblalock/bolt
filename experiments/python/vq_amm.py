#!/usr/bin/env python

import numpy as np

from . import vquantizers as vq
from . import amm

KEY_NLOOKUPS = 'nlookups'


class PQMatmul(amm.ApproxMatmul):

    def __init__(self, ncodebooks):
        self.ncodebooks = ncodebooks
        self.enc = self._create_encoder(ncodebooks)
        self._reset()

    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder(nsubvects=ncodebooks,
                            **self._get_encoder_kwargs())

    def _get_encoder_kwargs(self):  # to be overriden by subclasses
        return {}

    def _reset(self):
        self.A_enc = None
        self.luts = None

    def fit(self, A, B, Y=None):
        self.enc.fit(A, B)

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
        nmuls = 0
        nmuls += 0 if fixedA else A.shape[0] * A.shape[1] * 256   # enc cost
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * 256   # lut cost
        nlookups = A.shape[0] * B.shape[1] * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

    def get_params(self):
        return {'ncodebooks': self.ncodebooks}


class BoltMatmul(PQMatmul):

    def __init__(self, ncodebooks):
        self.ncodebooks = 2 * ncodebooks
        self.enc = self._create_encoder(self.ncodebooks)
        self._reset()

    def _create_encoder(self, ncodebooks):
        return vq.PQEncoder(nsubvects=ncodebooks, ncentroids=16,
                            # TODO set quantize_lut=True after debug
                            **self._get_encoder_kwargs())

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        nmuls = 0
        nmuls += 0 if fixedA else A.shape[0] * A.shape[1] * 16   # enc cost
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * 16   # lut cost
        nlookups = A.shape[0] * B.shape[1] * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}


class OPQMatmul(PQMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='OPQ')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        nmuls = 0
        nmuls += 0 if fixedA else A.shape[0] * A.shape[1] * 256   # enc cost
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * 256   # lut cost
        nmuls += A.shape[0] * A.shape[1] * A.shape[1]  # OPQ rotation cost
        nlookups = A.shape[0] * B.shape[1] * 2 * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}


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
