#!/usr/bin/env python

from __future__ import print_function

import pprint


microbench_output = \
"""
ncodebooks = 4
amm multisplit; N, D, M, ncodebooks:   9984, 512,  10,  4,   (5x5): 0.038 (2627368421/s), 0.036 (2773333333/s), 0.036 (2773333333/s), 0.035 (2852571428/s), 0.035 (2852571428/s),
amm multisplit; N, D, M, ncodebooks:   9984, 512, 100,  4,   (5x5): 0.212 (4709433962/s), 0.203 (4918226600/s), 0.204 (4894117647/s), 0.204 (4894117647/s), 0.21 (4754285714/s),
amm multisplit; N, D, M, ncodebooks:  57568,  24,   3,  4,   (5x5): 0.186 (928516129/s), 0.163 (1059533742/s), 0.159 (1086188679/s), 0.161 (1072695652/s), 0.16 (1079400000/s),
amm multisplit; N, D, M, ncodebooks: 115168,  24,   3,  4,   (5x5): 0.432 (799777777/s), 0.418 (826564593/s), 0.413 (836571428/s), 0.402 (859462686/s), 0.411 (840642335/s),
amm multisplit; N, D, M, ncodebooks: 230368,  24,   3,  4,   (5x5): 0.968 (713950413/s), 0.96 (719900000/s), 0.976 (708098360/s), 0.955 (723669109/s), 0.967 (714688728/s),
amm multisplit; N, D, M, ncodebooks:  49280,  27,   2,  4,   (5x5): 0.135 (730074074/s), 0.128 (770000000/s), 0.128 (770000000/s), 0.127 (776062992/s), 0.125 (788480000/s),
ncodebooks = 8
amm multisplit; N, D, M, ncodebooks:   9984, 512,  10,  8,   (5x5): 0.07 (1426285714/s), 0.067 (1490149253/s), 0.066 (1512727272/s), 0.065 (1536000000/s), 0.065 (1536000000/s),
amm multisplit; N, D, M, ncodebooks:   9984, 512, 100,  8,   (5x5): 0.314 (3179617834/s), 0.335 (2980298507/s), 0.321 (3110280373/s), 0.313 (3189776357/s), 0.336 (2971428571/s),
amm multisplit; N, D, M, ncodebooks:  57568,  24,   3,  8,   (5x5): 0.317 (544807570/s), 0.311 (555318327/s), 0.305 (566242622/s), 0.305 (566242622/s), 0.309 (558912621/s),
amm multisplit; N, D, M, ncodebooks: 115168,  24,   3,  8,   (5x5): 0.835 (413777245/s), 0.875 (394861714/s), 0.833 (414770708/s), 0.85 (406475294/s), 0.831 (415768953/s),
amm multisplit; N, D, M, ncodebooks: 230368,  24,   3,  8,   (5x5): 1.995 (346418045/s), 1.947 (354958397/s), 1.867 (370168184/s), 1.874 (368785485/s), 1.87 (369574331/s),
amm multisplit; N, D, M, ncodebooks:  49280,  27,   2,  8,   (5x5): 0.255 (386509803/s), 0.249 (395823293/s), 0.248 (397419354/s), 0.244 (403934426/s), 0.249 (395823293/s),
ncodebooks = 16
amm multisplit; N, D, M, ncodebooks:   9984, 512,  10, 16,   (5x5): 0.148 (674594594/s), 0.13 (768000000/s), 0.129 (773953488/s), 0.127 (786141732/s), 0.127 (786141732/s),
amm multisplit; N, D, M, ncodebooks:   9984, 512, 100, 16,   (5x5): 0.603 (1655721393/s), 0.56 (1782857142/s), 0.568 (1757746478/s), 0.562 (1776512455/s), 0.568 (1757746478/s),
amm multisplit; N, D, M, ncodebooks:  57568,  24,   3, 16,   (5x5): 0.703 (245667140/s), 0.664 (260096385/s), 0.657 (262867579/s), 0.652 (264883435/s), 0.658 (262468085/s),
amm multisplit; N, D, M, ncodebooks: 115168,  24,   3, 16,   (5x5): 1.689 (204561278/s), 1.67 (206888622/s), 1.668 (207136690/s), 1.673 (206517632/s), 1.702 (202998824/s),
amm multisplit; N, D, M, ncodebooks: 230368,  24,   3, 16,   (5x5): 4.024 (171745526/s), 3.792 (182253164/s), 4.041 (171023014/s), 3.849 (179554169/s), 3.97 (174081612/s),
amm multisplit; N, D, M, ncodebooks:  49280,  27,   2, 16,   (5x5): 0.559 (176314847/s), 0.55 (179200000/s), 0.556 (177266187/s), 0.536 (183880597/s), 0.542 (181845018/s),
ncodebooks = 32
amm multisplit; N, D, M, ncodebooks:   9984, 512,  10, 32,   (5x5): 0.384 (260000000/s), 0.284 (351549295/s), 0.278 (359136690/s), 0.275 (363054545/s), 0.278 (359136690/s),
amm multisplit; N, D, M, ncodebooks:   9984, 512, 100, 32,   (5x5): 1.178 (847538200/s), 1.107 (901897018/s), 1.071 (932212885/s), 1.077 (927019498/s), 1.086 (919337016/s),
amm multisplit; N, D, M, ncodebooks:  57568,  24,   3, 32,   (5x5): 1.392 (124068965/s), 1.36 (126988235/s), 1.375 (125602909/s), 1.395 (123802150/s), 1.422 (121451476/s),
amm multisplit; N, D, M, ncodebooks: 115168,  24,   3, 32,   (5x5): 3.668 (94194111/s), 3.643 (94840516/s), 3.623 (95364062/s), 3.629 (95206392/s), 3.57 (96779831/s),
amm multisplit; N, D, M, ncodebooks: 230368,  24,   3, 32,   (5x5): 7.751 (89163204/s), 7.407 (93304171/s), 7.878 (87725818/s), 7.713 (89602489/s), 7.315 (94477648/s),
amm multisplit; N, D, M, ncodebooks:  49280,  27,   2, 32,   (5x5): 1.089 (90505050/s), 1.114 (88473967/s), 1.069 (92198316/s), 1.058 (93156899/s), 1.051 (93777354/s),
matmul N, D, M:  10000, 512,  10,    (5x5): 5.682 (17599436/s), 5.612 (17818959/s), 5.73 (17452006/s), 5.754 (17379214/s), 5.767 (17340038/s),
matmul N, D, M:  10000, 512, 100,    (5x5): 14.7 (68027210/s), 14.649 (68264045/s), 14.539 (68780521/s), 14.525 (68846815/s), 14.512 (68908489/s),
matmul N, D, M:  57593,  24,   3,    (5x5): 0.448 (385667410/s), 0.453 (381410596/s), 0.453 (381410596/s), 0.441 (391789115/s), 0.442 (390902714/s),
matmul N, D, M: 115193,  24,   3,    (5x5): 1.012 (341481225/s), 1.012 (341481225/s), 1.01 (342157425/s), 1.012 (341481225/s), 1.017 (339802359/s),
matmul N, D, M: 230393,  24,   3,    (5x5): 2.352 (293868622/s), 2.359 (292996608/s), 2.36 (292872457/s), 2.365 (292253276/s), 2.358 (293120865/s),
matmul N, D, M:  49284,  27,   2,    (5x5): 0.473 (208389006/s), 0.441 (223510204/s), 0.385 (256020779/s), 0.378 (260761904/s), 0.383 (257357702/s),
"""


def load_matmul_times_for_n_d_m():
    lines = microbench_output.split('\n')
    matmul_lines = [line for line in lines if line.startswith('matmul')]
    matmul_shape_to_times = {}
    matmul_shape_to_thruputs = {}
    for line in matmul_lines:
        start_idx = line.find(':') + 1
        end_idx = line.find('(')
        nmd_str = line[start_idx:end_idx]
        N, D, M = [int(substr) for substr in nmd_str.split(',')[:3]]
        speeds_str = line[line.find('):') + 2:]
        speed_pairs = speeds_str.split(',')[:5]
        times = []
        thruputs = []
        for pair in speed_pairs:
            pair = pair.strip()
            time_str, thruput_str = pair.split()
            times.append(float(time_str))
            thruput_str = thruput_str.strip('()s/')
            thruputs.append(float(thruput_str))

        key = (N, D, M)
        matmul_shape_to_times[key] = times
        matmul_shape_to_thruputs[key] = thruputs

    pprint.pprint(matmul_shape_to_times)
    pprint.pprint(matmul_shape_to_thruputs)

    return matmul_shape_to_times, matmul_shape_to_thruputs


def load_multisplit_times_for_n_d_m():
    lines = microbench_output.split('\n')
    lines = [line for line in lines if line.startswith('amm multisplit')]
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
            time_str, thruput_str = pair.split()
            times.append(float(time_str))
            thruput_str = thruput_str.strip('()s/')
            thruputs.append(float(thruput_str))

        key = (N, D, M, C)
        shape_ncodebooks_to_times[key] = times
        shape_ncodebooks_to_thruputs[key] = thruputs

    pprint.pprint(shape_ncodebooks_to_times)
    pprint.pprint(shape_ncodebooks_to_thruputs)

    return shape_ncodebooks_to_times, shape_ncodebooks_to_thruputs


def main():
    load_matmul_times_for_n_d_m()
    load_multisplit_times_for_n_d_m()


if __name__ == '__main__':
    main()
