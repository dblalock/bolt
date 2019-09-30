#!/usr/bin/env python

import numpy as np

from . import vquantizers as vq
from . import amm

KEY_NLOOKUPS = 'nlookups'


class PQMatmul(amm.ApproxMatmul):

    def __init__(self, ncodebooks):
        self.ncodebooks = ncodebooks
        self.enc = vq.PQEncoder(nsubvects=ncodebooks)
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

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        nmuls = 0 if fixedB else B.shape[0] * B.shape[1] * 256
        return {amm.KEY_NMULTIPLIES: nmuls,
                KEY_NLOOKUPS: A.shape[0] * B.shape[1] * self.ncodebooks}

    def get_params(self):
        return {'ncodebooks': self.ncodebooks}
