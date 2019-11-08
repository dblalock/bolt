#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import pprint

microbench_output = \
"""
ncodebooks = 4
amm bolt N, D, M, ncodebooks:  10000, 512,  10,  4   (5x20): 7.574 (4.225e+07/s), 7.582 (4.221e+07/s), 7.584 (4.219e+07/s), 7.587 (4.218e+07/s), 7.579 (4.222e+07/s),
amm bolt N, D, M, ncodebooks:  10000, 512, 100,  4   (5x20): 7.747 (1.652e+08/s), 7.743 (1.653e+08/s), 7.757 (1.650e+08/s), 7.758 (1.650e+08/s), 7.743 (1.653e+08/s),
amm bolt N, D, M, ncodebooks: 223590,  96,  12,  4   (5x20): 26.029 (2.749e+08/s), 26.028 (2.749e+08/s), 26.013 (2.751e+08/s), 26.010 (2.751e+08/s), 26.063 (2.745e+08/s),
amm bolt N, D, M, ncodebooks:  49284,  27,   2,  4   (5x20): 1.931 (8.167e+08/s), 1.924 (8.197e+08/s), 1.925 (8.193e+08/s), 1.925 (8.193e+08/s), 1.929 (8.176e+08/s),
ncodebooks = 8
amm bolt N, D, M, ncodebooks:  10000, 512,  10,  8   (5x20): 6.912 (4.630e+07/s), 6.919 (4.625e+07/s), 6.912 (4.630e+07/s), 6.909 (4.632e+07/s), 6.911 (4.630e+07/s),
amm bolt N, D, M, ncodebooks:  10000, 512, 100,  8   (5x20): 7.169 (1.785e+08/s), 7.207 (1.776e+08/s), 7.200 (1.778e+08/s), 7.205 (1.777e+08/s), 7.205 (1.777e+08/s),
amm bolt N, D, M, ncodebooks: 223590,  96,  12,  8   (5x20): 24.550 (2.914e+08/s), 24.514 (2.919e+08/s), 24.485 (2.922e+08/s), 24.470 (2.924e+08/s), 24.474 (2.923e+08/s),
amm bolt N, D, M, ncodebooks:  49284,  27,   2,  8   (5x20): 2.445 (6.450e+08/s), 2.454 (6.427e+08/s), 2.436 (6.474e+08/s), 2.448 (6.442e+08/s), 2.446 (6.448e+08/s),
ncodebooks = 16
amm bolt N, D, M, ncodebooks:  10000, 512,  10, 16   (5x20): 6.350 (5.039e+07/s), 6.350 (5.039e+07/s), 6.347 (5.042e+07/s), 6.356 (5.035e+07/s), 6.438 (4.970e+07/s),
amm bolt N, D, M, ncodebooks:  10000, 512, 100, 16   (5x20): 7.340 (1.744e+08/s), 7.270 (1.761e+08/s), 7.302 (1.753e+08/s), 7.277 (1.759e+08/s), 7.299 (1.754e+08/s),
amm bolt N, D, M, ncodebooks: 223590,  96,  12, 16   (5x20): 28.217 (2.536e+08/s), 28.063 (2.550e+08/s), 28.082 (2.548e+08/s), 28.086 (2.547e+08/s), 28.070 (2.549e+08/s),
amm bolt N, D, M, ncodebooks:  49284,  27,   2, 16   (5x20): 3.525 (4.474e+08/s), 3.529 (4.469e+08/s), 3.525 (4.474e+08/s), 3.530 (4.468e+08/s), 3.527 (4.471e+08/s),
ncodebooks = 32
amm bolt N, D, M, ncodebooks:  10000, 512,  10, 32   (5x20): 6.036 (5.302e+07/s), 6.070 (5.272e+07/s), 6.085 (5.259e+07/s), 6.158 (5.196e+07/s), 6.176 (5.181e+07/s),
amm bolt N, D, M, ncodebooks:  10000, 512, 100, 32   (5x20): 7.473 (1.713e+08/s), 7.478 (1.712e+08/s), 7.571 (1.691e+08/s), 7.567 (1.692e+08/s), 7.571 (1.691e+08/s),
amm bolt N, D, M, ncodebooks: 223590,  96,  12, 32   (5x20): 36.693 (1.950e+08/s), 36.721 (1.948e+08/s), 36.753 (1.947e+08/s), 36.805 (1.944e+08/s), 37.216 (1.923e+08/s),
ncodebooks = 64
amm bolt N, D, M, ncodebooks:  10000, 512,  10, 64   (5x20): 6.962 (4.596e+07/s), 6.959 (4.598e+07/s), 6.954 (4.602e+07/s), 6.959 (4.598e+07/s), 6.964 (4.595e+07/s),
amm bolt N, D, M, ncodebooks:  10000, 512, 100, 64   (5x20): 8.539 (1.499e+08/s), 8.598 (1.489e+08/s), 8.484 (1.509e+08/s), 8.572 (1.493e+08/s), 8.527 (1.501e+08/s),
amm bolt N, D, M, ncodebooks: 223590,  96,  12, 64   (5x20): 64.087 (1.116e+08/s), 64.096 (1.116e+08/s), 64.638 (1.107e+08/s), 64.099 (1.116e+08/s), 64.079 (1.117e+08/s),
ncodebooks = 4
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512,  10,  4   (5x20): 0.021 (4.770e+09/s), 0.021 (4.770e+09/s), 0.021 (4.770e+09/s), 0.021 (4.770e+09/s), 0.021 (4.770e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512,  10,  4   (5x20): 0.016 (1.252e+09/s), 0.016 (1.252e+09/s), 0.016 (1.252e+09/s), 0.016 (1.252e+09/s), 0.016 (1.252e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512,  10,  4   (5x20): 0.000 (inf/s), 0.000 (inf/s), 0.000 (inf/s), 0.000 (inf/s), 0.000 (inf/s),
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512, 100,  4   (5x20): 0.077 (1.301e+10/s), 0.077 (1.301e+10/s), 0.076 (1.318e+10/s), 0.080 (1.252e+10/s), 0.077 (1.301e+10/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512, 100,  4   (5x20): 0.016 (1.252e+09/s), 0.016 (1.252e+09/s), 0.016 (1.252e+09/s), 0.016 (1.252e+09/s), 0.017 (1.178e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512, 100,  4   (5x20): 0.000 (inf/s), 0.000 (inf/s), 0.000 (inf/s), 0.000 (inf/s), 0.000 (inf/s),
----
f32 amm mithral      N, D, M, ncodebooks: 223590,  96,  12,  4   (5x20): 0.999 (2.686e+09/s), 0.974 (2.755e+09/s), 1.001 (2.681e+09/s), 1.000 (2.683e+09/s), 0.999 (2.686e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12,  4   (5x20): 0.614 (7.284e+08/s), 0.611 (7.320e+08/s), 0.598 (7.479e+08/s), 0.613 (7.296e+08/s), 0.601 (7.441e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12,  4   (5x20): 0.024 (1.863e+10/s), 0.024 (1.863e+10/s), 0.024 (1.863e+10/s), 0.024 (1.863e+10/s), 0.024 (1.863e+10/s),
----
i16 amm mithral      N, D, M, ncodebooks: 223590,  96,  12,  4   (5x20): 0.604 (4.443e+09/s), 0.603 (4.450e+09/s), 0.579 (4.635e+09/s), 0.604 (4.443e+09/s), 0.605 (4.435e+09/s),
i16 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12,  4   (5x20): 0.257 (1.740e+09/s), 0.280 (1.597e+09/s), 0.265 (1.688e+09/s), 0.254 (1.761e+09/s), 0.254 (1.761e+09/s),
i16 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12,  4   (5x20): 0.024 (1.863e+10/s), 0.024 (1.863e+10/s), 0.024 (1.863e+10/s), 0.024 (1.863e+10/s), 0.024 (1.863e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks:  49284,  27,   2,  4   (5x20): 0.083 (1.188e+09/s), 0.083 (1.188e+09/s), 0.085 (1.160e+09/s), 0.084 (1.174e+09/s), 0.084 (1.174e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2,  4   (5x20): 0.077 (1.281e+09/s), 0.077 (1.281e+09/s), 0.076 (1.298e+09/s), 0.076 (1.298e+09/s), 0.076 (1.298e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2,  4   (5x20): 0.004 (2.466e+10/s), 0.004 (2.466e+10/s), 0.004 (2.466e+10/s), 0.004 (2.466e+10/s), 0.004 (2.466e+10/s),
----
i8 amm mithral      N, D, M, ncodebooks:  49284,  27,   2,  4    (5x20): 0.034 (2.901e+09/s), 0.029 (3.401e+09/s), 0.029 (3.401e+09/s), 0.030 (3.287e+09/s), 0.030 (3.287e+09/s),
i8 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2,  4    (5x20): 0.023 (4.288e+09/s), 0.023 (4.288e+09/s), 0.023 (4.288e+09/s), 0.023 (4.288e+09/s), 0.023 (4.288e+09/s),
i8 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2,  4    (5x20): 0.004 (2.466e+10/s), 0.004 (2.466e+10/s), 0.004 (2.466e+10/s), 0.004 (2.466e+10/s), 0.004 (2.466e+10/s),
ncodebooks = 8
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512,  10,  8   (5x20): 0.043 (2.329e+09/s), 0.043 (2.329e+09/s), 0.043 (2.329e+09/s), 0.043 (2.329e+09/s), 0.043 (2.329e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512,  10,  8   (5x20): 0.031 (1.292e+09/s), 0.032 (1.252e+09/s), 0.033 (1.214e+09/s), 0.034 (1.178e+09/s), 0.034 (1.178e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512,  10,  8   (5x20): 0.001 (4.006e+10/s), 0.001 (4.006e+10/s), 0.001 (4.006e+10/s), 0.001 (4.006e+10/s), 0.001 (4.006e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512, 100,  8   (5x20): 0.154 (6.504e+09/s), 0.162 (6.183e+09/s), 0.155 (6.462e+09/s), 0.155 (6.462e+09/s), 0.162 (6.183e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512, 100,  8   (5x20): 0.035 (1.145e+09/s), 0.033 (1.214e+09/s), 0.032 (1.252e+09/s), 0.034 (1.178e+09/s), 0.034 (1.178e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512, 100,  8   (5x20): 0.001 (4.006e+10/s), 0.001 (4.006e+10/s), 0.001 (4.006e+10/s), 0.001 (4.006e+10/s), 0.001 (4.006e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks: 223590,  96,  12,  8   (5x20): 1.810 (1.483e+09/s), 1.790 (1.499e+09/s), 1.797 (1.493e+09/s), 1.809 (1.483e+09/s), 1.810 (1.483e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12,  8   (5x20): 1.395 (6.412e+08/s), 1.371 (6.524e+08/s), 1.394 (6.417e+08/s), 1.394 (6.417e+08/s), 1.393 (6.421e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12,  8   (5x20): 0.041 (2.182e+10/s), 0.041 (2.182e+10/s), 0.041 (2.182e+10/s), 0.041 (2.182e+10/s), 0.041 (2.182e+10/s),
----
i16 amm mithral      N, D, M, ncodebooks: 223590,  96,  12,  8   (5x20): 1.102 (2.435e+09/s), 1.106 (2.426e+09/s), 1.091 (2.460e+09/s), 1.101 (2.437e+09/s), 1.129 (2.377e+09/s),
i16 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12,  8   (5x20): 0.681 (1.313e+09/s), 0.653 (1.370e+09/s), 0.654 (1.368e+09/s), 0.653 (1.370e+09/s), 0.653 (1.370e+09/s),
i16 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12,  8   (5x20): 0.041 (2.182e+10/s), 0.041 (2.182e+10/s), 0.041 (2.182e+10/s), 0.043 (2.080e+10/s), 0.043 (2.080e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks:  49284,  27,   2,  8   (5x20): 0.173 (5.701e+08/s), 0.172 (5.734e+08/s), 0.173 (5.701e+08/s), 0.185 (5.331e+08/s), 0.173 (5.701e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2,  8   (5x20): 0.160 (1.233e+09/s), 0.176 (1.121e+09/s), 0.185 (1.066e+09/s), 0.165 (1.195e+09/s), 0.161 (1.225e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2,  8   (5x20): 0.008 (2.466e+10/s), 0.008 (2.466e+10/s), 0.008 (2.466e+10/s), 0.008 (2.466e+10/s), 0.008 (2.466e+10/s),
----
i8 amm mithral      N, D, M, ncodebooks:  49284,  27,   2,  8    (5x20): 0.059 (1.672e+09/s), 0.059 (1.672e+09/s), 0.059 (1.672e+09/s), 0.059 (1.672e+09/s), 0.059 (1.672e+09/s),
i8 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2,  8    (5x20): 0.049 (4.025e+09/s), 0.050 (3.945e+09/s), 0.049 (4.025e+09/s), 0.048 (4.109e+09/s), 0.048 (4.109e+09/s),
i8 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2,  8    (5x20): 0.008 (2.466e+10/s), 0.008 (2.466e+10/s), 0.008 (2.466e+10/s), 0.008 (2.466e+10/s), 0.008 (2.466e+10/s),
ncodebooks = 16
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512,  10, 16   (5x20): 0.094 (1.066e+09/s), 0.093 (1.077e+09/s), 0.100 (1.002e+09/s), 0.100 (1.002e+09/s), 0.097 (1.033e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512,  10, 16   (5x20): 0.065 (1.233e+09/s), 0.066 (1.214e+09/s), 0.066 (1.214e+09/s), 0.065 (1.233e+09/s), 0.066 (1.214e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512,  10, 16   (5x20): 0.003 (2.671e+10/s), 0.003 (2.671e+10/s), 0.003 (2.671e+10/s), 0.003 (2.671e+10/s), 0.003 (2.671e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512, 100, 16   (5x20): 0.367 (2.729e+09/s), 0.372 (2.692e+09/s), 0.374 (2.678e+09/s), 0.377 (2.657e+09/s), 0.374 (2.678e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512, 100, 16   (5x20): 0.067 (1.196e+09/s), 0.064 (1.252e+09/s), 0.064 (1.252e+09/s), 0.064 (1.252e+09/s), 0.064 (1.252e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512, 100, 16   (5x20): 0.003 (2.671e+10/s), 0.003 (2.671e+10/s), 0.003 (2.671e+10/s), 0.003 (2.671e+10/s), 0.003 (2.671e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks: 223590,  96,  12, 16   (5x20): 3.597 (7.460e+08/s), 3.607 (7.439e+08/s), 3.599 (7.456e+08/s), 3.610 (7.433e+08/s), 3.614 (7.425e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12, 16   (5x20): 2.761 (6.479e+08/s), 2.761 (6.479e+08/s), 2.760 (6.482e+08/s), 2.751 (6.503e+08/s), 2.763 (6.475e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12, 16   (5x20): 0.103 (1.737e+10/s), 0.105 (1.704e+10/s), 0.123 (1.454e+10/s), 0.128 (1.398e+10/s), 0.123 (1.454e+10/s),
----
i16 amm mithral      N, D, M, ncodebooks: 223590,  96,  12, 16   (5x20): 2.233 (1.202e+09/s), 2.261 (1.187e+09/s), 2.207 (1.216e+09/s), 2.207 (1.216e+09/s), 2.261 (1.187e+09/s),
i16 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12, 16   (5x20): 1.417 (1.262e+09/s), 1.563 (1.145e+09/s), 1.514 (1.182e+09/s), 1.464 (1.222e+09/s), 1.483 (1.206e+09/s),
i16 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12, 16   (5x20): 0.136 (1.315e+10/s), 0.130 (1.376e+10/s), 0.147 (1.217e+10/s), 0.133 (1.345e+10/s), 0.134 (1.335e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks:  49284,  27,   2, 16   (5x20): 0.397 (2.484e+08/s), 0.407 (2.423e+08/s), 0.395 (2.497e+08/s), 0.388 (2.542e+08/s), 0.385 (2.562e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2, 16   (5x20): 0.369 (1.069e+09/s), 0.368 (1.072e+09/s), 0.377 (1.046e+09/s), 0.375 (1.052e+09/s), 0.408 (9.669e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2, 16   (5x20): 0.019 (2.076e+10/s), 0.019 (2.076e+10/s), 0.019 (2.076e+10/s), 0.019 (2.076e+10/s), 0.019 (2.076e+10/s),
----
i8 amm mithral      N, D, M, ncodebooks:  49284,  27,   2, 16    (5x20): 0.131 (7.529e+08/s), 0.131 (7.529e+08/s), 0.131 (7.529e+08/s), 0.131 (7.529e+08/s), 0.131 (7.529e+08/s),
i8 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2, 16    (5x20): 0.103 (3.830e+09/s), 0.103 (3.830e+09/s), 0.103 (3.830e+09/s), 0.103 (3.830e+09/s), 0.104 (3.793e+09/s),
i8 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2, 16    (5x20): 0.019 (2.076e+10/s), 0.019 (2.076e+10/s), 0.019 (2.076e+10/s), 0.019 (2.076e+10/s), 0.019 (2.076e+10/s),
ncodebooks = 32
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512,  10, 32   (5x20): 0.201 (4.983e+08/s), 0.194 (5.163e+08/s), 0.205 (4.886e+08/s), 0.201 (4.983e+08/s), 0.200 (5.008e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512,  10, 32   (5x20): 0.142 (1.129e+09/s), 0.143 (1.121e+09/s), 0.144 (1.113e+09/s), 0.142 (1.129e+09/s), 0.161 (9.954e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512,  10, 32   (5x20): 0.007 (2.289e+10/s), 0.007 (2.289e+10/s), 0.007 (2.289e+10/s), 0.007 (2.289e+10/s), 0.007 (2.289e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512, 100, 32   (5x20): 0.762 (1.314e+09/s), 0.781 (1.282e+09/s), 0.756 (1.325e+09/s), 0.753 (1.330e+09/s), 0.798 (1.255e+09/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512, 100, 32   (5x20): 0.183 (8.757e+08/s), 0.149 (1.076e+09/s), 0.154 (1.041e+09/s), 0.150 (1.068e+09/s), 0.147 (1.090e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512, 100, 32   (5x20): 0.007 (2.289e+10/s), 0.007 (2.289e+10/s), 0.007 (2.289e+10/s), 0.007 (2.289e+10/s), 0.007 (2.289e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks: 223590,  96,  12, 32   (5x20): 7.958 (3.372e+08/s), 7.142 (3.757e+08/s), 7.148 (3.754e+08/s), 7.114 (3.772e+08/s), 7.135 (3.761e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12, 32   (5x20): 5.589 (6.402e+08/s), 5.642 (6.341e+08/s), 5.563 (6.432e+08/s), 5.592 (6.398e+08/s), 5.579 (6.413e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12, 32   (5x20): 0.341 (1.049e+10/s), 0.330 (1.084e+10/s), 0.327 (1.094e+10/s), 0.327 (1.094e+10/s), 0.328 (1.091e+10/s),
----
i16 amm mithral      N, D, M, ncodebooks: 223590,  96,  12, 32   (5x20): 4.369 (6.142e+08/s), 4.357 (6.159e+08/s), 4.537 (5.914e+08/s), 4.361 (6.153e+08/s), 4.406 (6.090e+08/s),
i16 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12, 32   (5x20): 2.888 (1.239e+09/s), 2.889 (1.238e+09/s), 2.898 (1.235e+09/s), 2.898 (1.235e+09/s), 2.909 (1.230e+09/s),
i16 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12, 32   (5x20): 0.329 (1.087e+10/s), 0.326 (1.098e+10/s), 0.331 (1.081e+10/s), 0.328 (1.091e+10/s), 0.345 (1.037e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks:  49284,  27,   2, 32   (5x20): 0.781 (1.263e+08/s), 0.785 (1.256e+08/s), 0.793 (1.244e+08/s), 0.788 (1.252e+08/s), 0.787 (1.253e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2, 32   (5x20): 0.814 (9.693e+08/s), 0.828 (9.529e+08/s), 0.755 (1.045e+09/s), 0.766 (1.030e+09/s), 0.768 (1.027e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2, 32   (5x20): 0.045 (1.753e+10/s), 0.041 (1.924e+10/s), 0.041 (1.924e+10/s), 0.046 (1.715e+10/s), 0.041 (1.924e+10/s),
----
i8 amm mithral      N, D, M, ncodebooks:  49284,  27,   2, 32    (5x20): 0.320 (3.082e+08/s), 0.303 (3.255e+08/s), 0.301 (3.277e+08/s), 0.321 (3.072e+08/s), 0.301 (3.277e+08/s),
i8 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2, 32    (5x20): 0.279 (2.828e+09/s), 0.260 (3.035e+09/s), 0.263 (3.000e+09/s), 0.221 (3.570e+09/s), 0.242 (3.260e+09/s),
i8 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2, 32    (5x20): 0.061 (1.293e+10/s), 0.044 (1.793e+10/s), 0.041 (1.924e+10/s), 0.041 (1.924e+10/s), 0.040 (1.972e+10/s),
ncodebooks = 64
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512,  10, 64   (5x20): 0.454 (2.206e+08/s), 0.497 (2.015e+08/s), 0.489 (2.048e+08/s), 0.486 (2.061e+08/s), 0.457 (2.192e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512,  10, 64   (5x20): 0.349 (9.184e+08/s), 0.344 (9.317e+08/s), 0.385 (8.325e+08/s), 0.377 (8.502e+08/s), 0.344 (9.317e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512,  10, 64   (5x20): 0.019 (1.687e+10/s), 0.019 (1.687e+10/s), 0.019 (1.687e+10/s), 0.019 (1.687e+10/s), 0.020 (1.603e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks:  10000, 512, 100, 64   (5x20): 1.586 (6.315e+08/s), 1.530 (6.546e+08/s), 1.531 (6.542e+08/s), 1.529 (6.551e+08/s), 1.539 (6.508e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks:  10000, 512, 100, 64   (5x20): 0.405 (7.914e+08/s), 0.408 (7.856e+08/s), 0.449 (7.138e+08/s), 0.403 (7.953e+08/s), 0.411 (7.798e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks:  10000, 512, 100, 64   (5x20): 0.020 (1.603e+10/s), 0.020 (1.603e+10/s), 0.019 (1.687e+10/s), 0.019 (1.687e+10/s), 0.019 (1.687e+10/s),
----
f32 amm mithral      N, D, M, ncodebooks: 223590,  96,  12, 64   (5x20): 14.943 (1.796e+08/s), 15.205 (1.765e+08/s), 14.912 (1.799e+08/s), 14.951 (1.795e+08/s), 14.981 (1.791e+08/s),
f32 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12, 64   (5x20): 11.376 (6.290e+08/s), 11.305 (6.330e+08/s), 11.313 (6.325e+08/s), 11.315 (6.324e+08/s), 11.312 (6.326e+08/s),
f32 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12, 64   (5x20): 0.877 (8.159e+09/s), 0.822 (8.705e+09/s), 0.845 (8.468e+09/s), 0.849 (8.428e+09/s), 0.836 (8.559e+09/s),
----
i16 amm mithral      N, D, M, ncodebooks: 223590,  96,  12, 64   (5x20): 9.459 (2.837e+08/s), 9.458 (2.837e+08/s), 9.420 (2.849e+08/s), 9.457 (2.837e+08/s), 9.452 (2.839e+08/s),
i16 amm mithral enc  N, D, M, ncodebooks: 223590,  96,  12, 64   (5x20): 5.819 (1.230e+09/s), 5.820 (1.230e+09/s), 5.824 (1.229e+09/s), 5.845 (1.224e+09/s), 5.901 (1.213e+09/s),
i16 amm mithral zipb N, D, M, ncodebooks: 223590,  96,  12, 64   (5x20): 0.818 (8.748e+09/s), 0.823 (8.695e+09/s), 0.803 (8.911e+09/s), 0.818 (8.748e+09/s), 0.851 (8.409e+09/s),
----
f32 amm mithral      N, D, M, ncodebooks:  49284,  27,   2, 64   (5x20): 1.571 (6.278e+07/s), 1.571 (6.278e+07/s), 1.573 (6.270e+07/s), 1.574 (6.266e+07/s), 1.571 (6.278e+07/s),
f32 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2, 64   (5x20): 1.479 (1.067e+09/s), 1.473 (1.071e+09/s), 1.475 (1.070e+09/s), 1.476 (1.069e+09/s), 1.473 (1.071e+09/s),
f32 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2, 64   (5x20): 0.114 (1.384e+10/s), 0.115 (1.372e+10/s), 0.115 (1.372e+10/s), 0.110 (1.435e+10/s), 0.115 (1.372e+10/s),
----
i8 amm mithral      N, D, M, ncodebooks:  49284,  27,   2, 64    (5x20): 0.561 (1.758e+08/s), 0.560 (1.761e+08/s), 0.561 (1.758e+08/s), 0.560 (1.761e+08/s), 0.560 (1.761e+08/s),
i8 amm mithral enc  N, D, M, ncodebooks:  49284,  27,   2, 64    (5x20): 0.453 (3.483e+09/s), 0.492 (3.207e+09/s), 0.470 (3.357e+09/s), 0.464 (3.401e+09/s), 0.494 (3.194e+09/s),
i8 amm mithral zipb N, D, M, ncodebooks:  49284,  27,   2, 64    (5x20): 0.114 (1.384e+10/s), 0.120 (1.315e+10/s), 0.116 (1.360e+10/s), 0.114 (1.384e+10/s), 0.114 (1.384e+10/s),
blas sketch matmul N, D, M, d:  10000, 512,  10,   2     (5x20): 3.827 (2.613e+07/s), 3.815 (2.621e+07/s), 3.830 (2.611e+07/s), 3.858 (2.592e+07/s), 3.832 (2.610e+07/s),
our  sketch matmul N, D, M, d:  10000, 512,  10,   2     (5x20): 1.080 (9.259e+07/s), 1.041 (9.606e+07/s), 1.049 (9.533e+07/s), 1.049 (9.533e+07/s), 1.045 (9.569e+07/s),
blas sketch matmul N, D, M, d:  10000, 512,  10,   4     (5x20): 3.505 (2.853e+07/s), 3.568 (2.803e+07/s), 3.541 (2.824e+07/s), 3.431 (2.915e+07/s), 3.234 (3.092e+07/s),
our  sketch matmul N, D, M, d:  10000, 512,  10,   4     (5x20): 2.081 (4.805e+07/s), 2.135 (4.684e+07/s), 2.083 (4.801e+07/s), 2.077 (4.815e+07/s), 2.079 (4.810e+07/s),
blas sketch matmul N, D, M, d:  10000, 512,  10,   8     (5x20): 3.772 (2.651e+07/s), 3.641 (2.746e+07/s), 3.617 (2.765e+07/s), 3.616 (2.765e+07/s), 4.002 (2.499e+07/s),
our  sketch matmul N, D, M, d:  10000, 512,  10,   8     (5x20): 2.864 (3.492e+07/s), 2.861 (3.495e+07/s), 2.901 (3.447e+07/s), 3.017 (3.315e+07/s), 2.880 (3.472e+07/s),
blas sketch matmul N, D, M, d:  10000, 512,  10,  16     (5x20): 4.535 (2.205e+07/s), 4.565 (2.191e+07/s), 4.475 (2.235e+07/s), 4.476 (2.234e+07/s), 4.480 (2.232e+07/s),
our  sketch matmul N, D, M, d:  10000, 512,  10,  16     (5x20): 5.217 (1.917e+07/s), 5.185 (1.929e+07/s), 5.243 (1.907e+07/s), 5.256 (1.903e+07/s), 5.184 (1.929e+07/s),
blas sketch matmul N, D, M, d:  10000, 512,  10,  32     (5x20): 6.537 (1.530e+07/s), 6.527 (1.532e+07/s), 6.517 (1.534e+07/s), 6.507 (1.537e+07/s), 6.512 (1.536e+07/s),
our  sketch matmul N, D, M, d:  10000, 512,  10,  32     (5x20): 9.143 (1.094e+07/s), 9.119 (1.097e+07/s), 9.137 (1.094e+07/s), 9.110 (1.098e+07/s), 9.128 (1.096e+07/s),
blas sketch matmul N, D, M, d:  10000, 512,  10,  64     (5x20): 10.156 (9.846e+06/s), 10.136 (9.866e+06/s), 10.143 (9.859e+06/s), 10.146 (9.856e+06/s), 10.147 (9.855e+06/s),
our  sketch matmul N, D, M, d:  10000, 512,  10,  64     (5x20): 17.739 (5.637e+06/s), 17.767 (5.628e+06/s), 17.641 (5.669e+06/s), 17.647 (5.667e+06/s), 17.640 (5.669e+06/s),
blas sketch matmul N, D, M, d:  10000, 512,  10, 128     (5x20): 17.149 (5.831e+06/s), 17.183 (5.820e+06/s), 17.144 (5.833e+06/s), 17.109 (5.845e+06/s), 17.182 (5.820e+06/s),
our  sketch matmul N, D, M, d:  10000, 512,  10, 128     (5x20): 35.289 (2.834e+06/s), 35.025 (2.855e+06/s), 35.294 (2.833e+06/s), 35.022 (2.855e+06/s), 35.071 (2.851e+06/s),
blas matmul N, D, M:  10000, 512,  10    (5x20): 4.174 (2.396e+07/s), 4.136 (2.418e+07/s), 4.164 (2.402e+07/s), 4.198 (2.382e+07/s), 4.188 (2.388e+07/s),
our  matmul N, D, M:  10000, 512,  10    (5x20): 3.546 (2.820e+07/s), 3.546 (2.820e+07/s), 3.553 (2.815e+07/s), 3.555 (2.813e+07/s), 3.560 (2.809e+07/s),
blas sketch matmul N, D, M, d:  10000, 512, 100,   2     (5x20): 4.085 (2.448e+08/s), 4.091 (2.444e+08/s), 4.055 (2.466e+08/s), 4.045 (2.472e+08/s), 4.057 (2.465e+08/s),
our  sketch matmul N, D, M, d:  10000, 512, 100,   2     (5x20): 1.322 (7.564e+08/s), 1.337 (7.479e+08/s), 1.336 (7.485e+08/s), 1.323 (7.559e+08/s), 1.322 (7.564e+08/s),
blas sketch matmul N, D, M, d:  10000, 512, 100,   4     (5x20): 3.631 (2.754e+08/s), 3.843 (2.602e+08/s), 3.798 (2.633e+08/s), 3.848 (2.599e+08/s), 3.847 (2.599e+08/s),
our  sketch matmul N, D, M, d:  10000, 512, 100,   4     (5x20): 2.626 (3.808e+08/s), 2.491 (4.014e+08/s), 2.510 (3.984e+08/s), 2.589 (3.862e+08/s), 2.480 (4.032e+08/s),
blas sketch matmul N, D, M, d:  10000, 512, 100,   8     (5x20): 4.275 (2.339e+08/s), 4.313 (2.319e+08/s), 4.333 (2.308e+08/s), 4.289 (2.332e+08/s), 4.130 (2.421e+08/s),
our  sketch matmul N, D, M, d:  10000, 512, 100,   8     (5x20): 3.405 (2.937e+08/s), 3.571 (2.800e+08/s), 3.405 (2.937e+08/s), 3.423 (2.921e+08/s), 3.405 (2.937e+08/s),
blas sketch matmul N, D, M, d:  10000, 512, 100,  16     (5x20): 5.392 (1.855e+08/s), 5.316 (1.881e+08/s), 5.283 (1.893e+08/s), 5.281 (1.894e+08/s), 5.184 (1.929e+08/s),
our  sketch matmul N, D, M, d:  10000, 512, 100,  16     (5x20): 6.046 (1.654e+08/s), 6.047 (1.654e+08/s), 6.076 (1.646e+08/s), 6.071 (1.647e+08/s), 6.044 (1.655e+08/s),
blas sketch matmul N, D, M, d:  10000, 512, 100,  32     (5x20): 7.291 (1.372e+08/s), 7.293 (1.371e+08/s), 7.308 (1.368e+08/s), 7.296 (1.371e+08/s), 7.294 (1.371e+08/s),
our  sketch matmul N, D, M, d:  10000, 512, 100,  32     (5x20): 10.697 (9.348e+07/s), 10.584 (9.448e+07/s), 10.599 (9.435e+07/s), 10.611 (9.424e+07/s), 10.594 (9.439e+07/s),
blas sketch matmul N, D, M, d:  10000, 512, 100,  64     (5x20): 11.586 (8.631e+07/s), 11.528 (8.675e+07/s), 11.528 (8.675e+07/s), 11.535 (8.669e+07/s), 11.530 (8.673e+07/s),
our  sketch matmul N, D, M, d:  10000, 512, 100,  64     (5x20): 20.459 (4.888e+07/s), 20.514 (4.875e+07/s), 20.542 (4.868e+07/s), 20.429 (4.895e+07/s), 20.532 (4.870e+07/s),
blas matmul N, D, M:  10000, 512, 100    (5x20): 13.506 (7.404e+07/s), 13.432 (7.445e+07/s), 13.467 (7.426e+07/s), 13.464 (7.427e+07/s), 13.484 (7.416e+07/s),
our  matmul N, D, M:  10000, 512, 100    (5x20): 27.160 (3.682e+07/s), 27.135 (3.685e+07/s), 27.260 (3.668e+07/s), 27.213 (3.675e+07/s), 27.268 (3.667e+07/s),
blas sketch matmul N, D, M, d: 223590,  96,  12,   2     (5x20): 17.987 (1.492e+08/s), 17.601 (1.524e+08/s), 18.118 (1.481e+08/s), 17.847 (1.503e+08/s), 17.977 (1.493e+08/s),
our  sketch matmul N, D, M, d: 223590,  96,  12,   2     (5x20): 5.117 (5.243e+08/s), 5.115 (5.246e+08/s), 5.102 (5.259e+08/s), 5.088 (5.273e+08/s), 5.111 (5.250e+08/s),
blas sketch matmul N, D, M, d: 223590,  96,  12,   4     (5x20): 11.524 (2.328e+08/s), 12.362 (2.170e+08/s), 11.828 (2.268e+08/s), 11.793 (2.275e+08/s), 11.785 (2.277e+08/s),
our  sketch matmul N, D, M, d: 223590,  96,  12,   4     (5x20): 9.979 (2.689e+08/s), 10.007 (2.681e+08/s), 10.010 (2.680e+08/s), 10.010 (2.680e+08/s), 9.973 (2.690e+08/s),
blas sketch matmul N, D, M, d: 223590,  96,  12,   8     (5x20): 19.261 (1.393e+08/s), 19.116 (1.404e+08/s), 19.205 (1.397e+08/s), 19.342 (1.387e+08/s), 19.189 (1.398e+08/s),
our  sketch matmul N, D, M, d: 223590,  96,  12,   8     (5x20): 14.543 (1.845e+08/s), 14.510 (1.849e+08/s), 14.570 (1.842e+08/s), 14.556 (1.843e+08/s), 14.509 (1.849e+08/s),
blas matmul N, D, M: 223590,  96,  12    (5x20): 19.189 (1.398e+08/s), 19.231 (1.395e+08/s), 19.378 (1.385e+08/s), 19.348 (1.387e+08/s), 19.390 (1.384e+08/s),
our  matmul N, D, M: 223590,  96,  12    (5x20): 16.242 (1.652e+08/s), 16.194 (1.657e+08/s), 16.197 (1.657e+08/s), 16.230 (1.653e+08/s), 16.238 (1.652e+08/s),
blas sketch matmul N, D, M, d:  49284,  27,   2,   2     (5x20): 0.375 (2.628e+08/s), 0.373 (2.643e+08/s), 0.380 (2.594e+08/s), 0.380 (2.594e+08/s), 0.378 (2.608e+08/s),
our  sketch matmul N, D, M, d:  49284,  27,   2,   2     (5x20): 0.219 (4.501e+08/s), 0.220 (4.480e+08/s), 0.219 (4.501e+08/s), 0.216 (4.563e+08/s), 0.203 (4.856e+08/s),
blas matmul N, D, M:  49284,  27,   2    (5x20): 0.327 (3.014e+08/s), 0.318 (3.100e+08/s), 0.319 (3.090e+08/s), 0.328 (3.005e+08/s), 0.328 (3.005e+08/s),
our  matmul N, D, M:  49284,  27,   2    (5x20): 0.186 (5.299e+08/s), 0.181 (5.446e+08/s), 0.183 (5.386e+08/s), 0.174 (5.665e+08/s), 0.173 (5.698e+08/s),
"""


def _load_matmul_times_for_n_d_m(startswith):
    lines = microbench_output.split('\n')
    matmul_lines = [line for line in lines if line.startswith(startswith)]
    matmul_shape_to_times = {}
    matmul_shape_to_thruputs = {}
    for line in matmul_lines:
        start_idx = line.find(':') + 1
        end_idx = line.find('(')
        nmd_str = line[start_idx:end_idx]
        N, D, M = [int(substr) for substr in nmd_str.split(',')[:3]]
        speeds_str = line[line.find('):') + 2:]
        speed_pairs = speeds_str.split(',')[:5]
        # print("N, D, M: ", N, D, M)
        # print("speed pairs: ", speed_pairs)
        times = []
        thruputs = []
        for pair in speed_pairs:
            pair = pair.strip()
            if not len(pair):
                continue  # handle trailing comma on line
            # print("pair: ", pair)
            pair = pair.strip()
            time_str, thruput_str = pair.split()
            times.append(float(time_str))
            thruput_str = thruput_str.strip('()s/')
            thruputs.append(float(thruput_str))

        key = (N, D, M)
        matmul_shape_to_times[key] = times
        matmul_shape_to_thruputs[key] = thruputs

    # print("what we're getting from func:")
    # pprint.pprint(matmul_shape_to_times)
    # pprint.pprint(matmul_shape_to_thruputs)

    return matmul_shape_to_times, matmul_shape_to_thruputs


def _load_sketch_times_for_n_d_m(startswith):
    # print("loading sketch times for ", startswith)
    lines = microbench_output.split('\n')
    matmul_lines = [line for line in lines if line.startswith(startswith)]
    matmul_shape_to_times = {}
    matmul_shape_to_thruputs = {}
    for line in matmul_lines:
        start_idx = line.find(':') + 1
        end_idx = line.find('(')
        nmd_str = line[start_idx:end_idx]
        N, D, M, d = [int(substr) for substr in nmd_str.split(',')[:4]]
        speeds_str = line[line.find('):') + 2:]
        speed_pairs = speeds_str.split(',')[:5]
        # print("N, D, M: ", N, D, M)
        # print("speed pairs: ", speed_pairs)
        times = []
        thruputs = []
        for pair in speed_pairs:
            pair = pair.strip()
            if not len(pair):
                continue  # handle trailing comma on line
            # print("pair: ", pair)
            pair = pair.strip()
            time_str, thruput_str = pair.split()
            times.append(float(time_str))
            thruput_str = thruput_str.strip('()s/')
            thruputs.append(float(thruput_str))

        key = (N, D, M, d)
        matmul_shape_to_times[key] = times
        matmul_shape_to_thruputs[key] = thruputs

    # pprint.pprint(matmul_shape_to_times)
    # pprint.pprint(matmul_shape_to_thruputs)

    return matmul_shape_to_times, matmul_shape_to_thruputs


def load_matmul_times_for_n_d_m(key1='blas matmul', key2='our  matmul',
                                sketches=False):
    if sketches:
        # print("results from blas:")
        shape2lat0, shape2thruput0 = _load_sketch_times_for_n_d_m(key1)
        # print("results from ours:")
        shape2lat1, shape2thruput1 = _load_sketch_times_for_n_d_m(key2)
    else:
        # print("results from blas:")
        shape2lat0, shape2thruput0 = _load_matmul_times_for_n_d_m(key1)
        # print("results from ours:")
        shape2lat1, shape2thruput1 = _load_matmul_times_for_n_d_m(key2)

    # take minimum of time from eigen blas and our sgemm
    shape2lat = {}
    for k in shape2lat0:
        vals0 = shape2lat0.get(k, [1e20])
        vals1 = shape2lat1.get(k, [1e20])
        mean0, mean1 = np.mean(vals0), np.mean(vals1)
        if mean0 < mean1:
            shape2lat[k] = shape2lat0[k]
        else:
            shape2lat[k] = shape2lat1[k]
    shape2thruput = {}
    for k in shape2thruput0:
        vals0 = shape2thruput0.get(k, [-1e20])
        vals1 = shape2thruput1.get(k, [-1e20])
        # print("k, vals0, vals1: ", k)
        # print(vals0)
        # print(vals1)
        mean0, mean1 = np.mean(vals0), np.mean(vals1)
        if mean0 > mean1:
            shape2thruput[k] = shape2thruput0[k]
        else:
            shape2thruput[k] = shape2thruput1[k]

    # print("what we're returning:")
    # pprint.pprint(shape2lat)
    # pprint.pprint(shape2thruput)

    return shape2lat, shape2thruput


def _load_vq_times_for_n_d_m(startswith):
    lines = microbench_output.split('\n')
    lines = [line for line in lines if line.startswith(startswith)]
    shape_ncodebooks_to_times = {}
    shape_ncodebooks_to_thruputs = {}
    for line in lines:
        start_idx = line.find(':') + 1
        end_idx = line.find('(')
        nmd_str = line[start_idx:end_idx]
        N, D, M, C = [int(substr) for substr in nmd_str.split(',')[:4]]
        speeds_str = line[line.find('):') + 2:]
        speed_pairs = speeds_str.split(',')[:5]
        times = []
        thruputs = []
        for pair in speed_pairs:
            pair = pair.strip()
            if not len(pair):
                continue  # handle trailing comma on line
            time_str, thruput_str = pair.split()
            times.append(float(time_str))
            thruput_str = thruput_str.strip('()s/')
            thruputs.append(float(thruput_str))

        key = (N, D, M, C)
        shape_ncodebooks_to_times[key] = times
        shape_ncodebooks_to_thruputs[key] = thruputs

    # print("startswith: ", startswith)
    # if 'bolt' in startswith:
    #     print("bolt speed dicts:")
    #     pprint.pprint(shape_ncodebooks_to_times)
    #     pprint.pprint(shape_ncodebooks_to_thruputs)

    return shape_ncodebooks_to_times, shape_ncodebooks_to_thruputs


# def load_multisplit_times_for_n_d_m():
#     return _load_vq_times_for_n_d_m('famm mithral')


def load_bolt_times_for_n_d_m():
    return _load_vq_times_for_n_d_m('amm bolt')


def load_mithral_f32_times_for_n_d_m():
    # two spaces so it doesn't try to read enc and zip times
    return _load_vq_times_for_n_d_m('f32 amm mithral  ')


def load_mithral_i16_times_for_n_d_m():
    return _load_vq_times_for_n_d_m('i16 amm mithral  ')


def load_mithral_i8_times_for_n_d_m():
    return _load_vq_times_for_n_d_m('i8 amm mithral  ')


def load_mithral_times_for_n_d_m():
    return _load_vq_times_for_n_d_m('f32 amm mithral  ')


def load_sketch_times_for_n_d_m():
    return load_matmul_times_for_n_d_m(
        'blas sketch matmul', 'our  sketch matmul', sketches=True)


def main():
    # load_matmul_times_for_n_d_m()
    # load_multisplit_times_for_n_d_m()
    # load_bolt_times_for_n_d_m()
    # pprint.pprint(load_sketch_times_for_n_d_m())
    # pprint.pprint(load_multisplit_times_for_n_d_m())
    # pprint.pprint(load_mithral_times_for_n_d_m())
    ret = load_matmul_times_for_n_d_m()
    print("matmul latencies, thruputs")
    pprint.pprint(ret)
    # pprint.pprint(load_bolt_times_for_n_d_m())


if __name__ == '__main__':
    main()
