#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

# TODO this file is hideous (but necessarily so for deadline purposes...)
#
# Also, this file is tightly coupled to figs.py; it basically has a func
# for each figure func that spits out data in exactly the required form


MCQ_RESULTS_DIR = '../results/timing/'
MATMUL_RESULTS_DIR = '../results/matmul/'


def get_mcq_path(D, nbytes):
    fname = 'mcq_D={}_M={}.txt'.format(D, nbytes)
    return os.path.join(MCQ_RESULTS_DIR, fname)


class McqResults(object):

    def __init__(self, path=None, D=None, nbytes=None):

        if path is None:
            path = get_mcq_path(D=D, nbytes=nbytes)

        self.path = path

        with open(self.path, 'r') as f:
            self.lines = f.readlines()

        self.stats = {line.split(':')[0].strip(): line.split(':')[1].strip()
                      for line in self.lines if ':' in line}
        self.bolt_nbytes = int(self.stats['bolt M'])
        self.pq_nbytes = int(self.stats['pq M'])
        self.bolt_D = int(self.stats['bolt subvect_len']) * self.bolt_nbytes * 2
        self.pq_D = int(self.stats['pq subvect_len']) * self.pq_nbytes

        assert self.bolt_nbytes == self.pq_nbytes
        assert self.bolt_D == self.pq_D

        self.nbytes = self.bolt_nbytes
        self.D = self.bolt_D

        # check that file was named properly
        expected_path = get_mcq_path(D=self.D, nbytes=self.nbytes)
        if expected_path != path:
            print("expected path, path = ", expected_path, path)
            assert expected_path == path

    def __str__(self):  # for debugging
        s = ""
        sorted_keys = sorted(self.stats.keys())
        for k in sorted_keys:
            v = self.stats[k]
            s += "'{}': '{}'\n".format(k, v)
        return s


def _extract_thruput(profile_str):
    result_strs = profile_str.split(':')[-1]
    rep_strs = result_strs.strip(' ,').split(',')
    thruput_parens = [s.strip(' ').split(' ')[1] for s in rep_strs]
    return np.array([int(s.strip('()s/')) for s in thruput_parens])


def _extract_times(profile_str):
    result_strs = profile_str.split(':')[-1]
    rep_strs = result_strs.strip(' ,').split(',')
    time_strs = [s.strip(' ').split(' ')[0] for s in rep_strs]
    return np.array([float(s) for s in time_strs])


def popcount_results_256():
    LENGTH = 256

    popcnt_times = {}
    popcnt_times[8] = '2.456 (1302931596/s), 2.344 (1365187713/s), 2.125 (1505882352/s), 2.829 (1131141746/s), 2.148 (1489757914/s), 2.167 (1476695892/s), 2.327 (1375161151/s), 2.145 (1491841491/s), 2.12 (1509433962/s), 2.112 (1515151515/s)'
    popcnt_times[16] = '4.368 (732600732/s), 4.121 (776510555/s), 3.926 (815078960/s), 4.105 (779537149/s), 4.176 (766283524/s), 4.119 (776887594/s), 4.464 (716845878/s), 4.153 (770527329/s), 4.364 (733272227/s), 4.198 (762267746/s)'
    popcnt_times[32] = '7.612 (420388859/s), 7.347 (435551925/s), 7.694 (415908500/s), 9.122 (350800263/s), 7.343 (435789186/s), 9.344 (342465753/s), 8.148 (392734413/s), 9.046 (353747512/s), 8.455 (378474275/s), 7.685 (416395575/s)'

    bolt_times = {}
    bolt_times[8] = '0.461 (2169197396/s), 0.456 (2192982456/s), 0.539 (1855287569/s), 0.53 (1886792452/s), 0.456 (2192982456/s), 0.452 (2212389380/s), 0.442 (2262443438/s), 0.438 (2283105022/s), 0.434 (2304147465/s), 0.547 (1828153564/s)'
    bolt_times[16] = '0.894 (1118568232/s), 1.08 (925925925/s), 0.88 (1136363636/s), 0.877 (1140250855/s), 0.881 (1135073779/s), 0.847 (1180637544/s), 1.011 (989119683/s), 0.866 (1154734411/s), 0.984 (1016260162/s), 0.838 (1193317422/s)'
    bolt_times[32] = '2.047 (488519785/s), 1.726 (579374275/s), 1.924 (519750519/s), 2.085 (479616306/s), 2.076 (481695568/s), 1.748 (572082379/s), 1.757 (569151963/s), 2.064 (484496124/s), 1.742 (574052812/s), 1.725 (579710144/s)'

    out_dicts = []
    algos = ['Bolt', 'Binary Embedding']
    dicts = [bolt_times, popcnt_times]
    for algo, d in zip(algos, dicts):
        for nbytes, s in list(d.items()):
            thruputs = _extract_thruput(s)
            out_dicts += [{'algo': algo, 'nbytes': nbytes, 'length': LENGTH,
                          'trial': i, 'y': t} for i, t in enumerate(thruputs)]

    return pd.DataFrame.from_records(out_dicts)


def encode_results():
    dicts = []
    for D in [64, 128, 256, 512, 1024]:
        for nbytes in [8, 16, 32]:
            res = McqResults(D=D, nbytes=nbytes)

            abbrevs = ['bolt', 'pq', 'opq']
            names = ['Bolt', 'PQ', 'OPQ']
            for abbrev, name in zip(abbrevs, names):
                # results for encoding data
                key = abbrev + ' encode (10x5)'
                thruputs = _extract_thruput(res.stats[key])
                dicts += [{'task': 'encode_x', 'D': D, 'nbytes': nbytes,
                           'algo': name, 'trial': i, 'y': t}
                          for i, t in enumerate(thruputs)]

                # results for encoding query
                if abbrev == 'bolt':
                    key = abbrev + ' encode lut (10x5)'
                else:
                    key = abbrev + ' encode lut float dist (10x5)'
                thruputs = _extract_thruput(res.stats[key])
                dicts += [{'task': 'encode_q', 'D': D, 'nbytes': nbytes,
                           'algo': name, 'trial': i, 'y': t}
                          for i, t in enumerate(thruputs)]

    return pd.DataFrame.from_records(dicts)


def matmul_results(which='square'):
    if which == 'square':
        SIZES = [64, 128, 256, 512, 1024, 4096, 8192]
        data_fname = 'square_matmul_results.txt'
    elif which == 'tall':
        SIZES = [32, 64, 128, 256, 512, 1024]
        data_fname = 'tall_matmul_results.txt'

    with open(MATMUL_RESULTS_DIR + data_fname) as f:
        lines = f.readlines()

    stats = {line.split(':')[0].strip(): line.split(':')[1].strip()
             for line in lines if ':' in line}

    dicts = []

    # add in results from bolt
    for nbytes in [8, 16, 32]:
        prefix = 'bolt<{}>'.format(nbytes)
        algo = 'Bolt {}B'.format(nbytes)
        for sz in SIZES:
            for enc in (0, 1):  # don't vs do encode X at start
                key = '{} encode={} matmul {} (10x5)'.format(prefix, enc, sz)
                times = _extract_times(stats[key])
                dicts += [{'algo': algo, 'size': sz, 'enc': enc, 'nbytes': nbytes,
                           'trial': i, 'y': t}
                          for i, t in enumerate(times)]

                # also add in "encode" version of bolt
                if enc:
                    enc_algo_name = algo + ' + Encode'
                    dicts += [{'algo': enc_algo_name, 'size': sz, 'enc': enc,
                               'nbytes': nbytes, 'trial': i, 'y': t}
                              for i, t in enumerate(times)]

    # add in matmul results
    for sz in SIZES:
        key = 'matmul {} (10x5)'.format(sz)
        times = _extract_times(stats[key])
        dicts += [{'algo': 'Floats', 'size': sz, 'enc': -1, 'trial': i, 'y': t}
                  for i, t in enumerate(times)]

    return pd.DataFrame.from_records(dicts)


def encode_data_results_256():
    LENGTH = 256

    pq_times = {}
    pq_times[8] = 'pq encode (10x5): 6.696 (149342/s), 6.688 (149521/s), 6.639 (150625/s), 6.648 (150421/s), 6.711 (149009/s), 6.67 (149925/s), 6.634 (150738/s), 6.684 (149611/s), 6.663 (150082/s), 6.67 (149925/s),'
    pq_times[16] = 'pq encode (10x5): 7.181 (139256/s), 7.194 (139004/s), 7.179 (139295/s), 7.146 (139938/s), 7.123 (140390/s), 7.123 (140390/s), 7.162 (139625/s), 7.148 (139899/s), 7.116 (140528/s), 7.193 (139024/s),'
    pq_times[32] = 'pq encode (10x5): 8.089 (123624/s), 8.175 (122324/s), 8.117 (123198/s), 8.096 (123517/s), 8.48 (117924/s), 8.071 (123900/s), 8.126 (123061/s), 8.123 (123107/s), 8.069 (123931/s), 8.21 (121802/s),'

    opq_times = {}
    opq_times[8] = 'opq encode (10x5): 8.441 (118469/s), 8.385 (119260/s), 8.368 (119502/s), 8.39 (119189/s), 8.355 (119688/s), 8.388 (119217/s), 8.383 (119289/s), 8.412 (118877/s), 8.401 (119033/s), 8.391 (119175/s),'
    opq_times[16] = 'opq encode (10x5): 8.88 (112612/s), 8.786 (113817/s), 8.874 (112688/s), 8.834 (113199/s), 8.874 (112688/s), 8.902 (112334/s), 8.899 (112372/s), 8.925 (112044/s), 8.867 (112777/s), 8.907 (112271/s),'
    opq_times[32] = 'opq encode (10x5): 9.761 (102448/s), 9.718 (102901/s), 9.717 (102912/s), 9.726 (102817/s), 9.908 (100928/s), 9.796 (102082/s), 10.164 (98386/s), 9.792 (102124/s), 9.735 (102722/s), 9.729 (102785/s),'

    bolt_times = {}
    bolt_times[8] = 'bolt encode (10x5): 3.43 (2915451/s), 3.586 (2788622/s), 3.421 (2923121/s), 3.408 (2934272/s), 3.409 (2933411/s), 3.406 (2935995/s), 3.407 (2935133/s), 3.412 (2930832/s), 3.411 (2931691/s), 3.409 (2933411/s),'
    bolt_times[16] = 'bolt encode (10x5): 3.93 (2544529/s), 3.687 (2712232/s), 3.826 (2613695/s), 4.007 (2495632/s), 3.705 (2699055/s), 3.976 (2515090/s), 3.709 (2696144/s), 3.681 (2716653/s), 3.693 (2707825/s), 3.802 (2630194/s),'
    bolt_times[32] = 'bolt encode (10x5): 5.039 (1984520/s), 4.591 (2178174/s), 5.081 (1968116/s), 4.697 (2129018/s), 4.591 (2178174/s), 4.763 (2099517/s), 4.832 (2069536/s), 4.805 (2081165/s), 4.961 (2015722/s), 4.665 (2143622/s),'

    out_dicts = []
    algos = ['Bolt', 'PQ', 'OPQ']
    dicts = [bolt_times, pq_times, opq_times]
    for algo, d in zip(algos, dicts):
        for nbytes, s in list(d.items()):
            thruputs = _extract_thruput(s)
            out_dicts += [{'algo': algo, 'nbytes': nbytes, 'length': LENGTH,
                          'trial': i, 'y': t} for i, t in enumerate(thruputs)]

    return pd.DataFrame.from_records(out_dicts)


def encode_lut_results():
    pq_times = {}
    pq_times[8] = 'pq encode lut float dist (10x5): 64.986 (153879/s), 65.014 (153813/s), 65.155 (153480/s), 64.808 (154301/s), 66.593 (150165/s), 67.68 (147754/s), 69.399 (144094/s), 66.702 (149920/s), 66.234 (150979/s), 66.286 (150861/s),'
    pq_times[16] = 'pq encode lut float dist (10x5): 67.893 (147290/s), 67.484 (148183/s), 69.608 (143661/s), 68.083 (146879/s), 70.958 (140928/s), 69.423 (144044/s), 72.129 (138640/s), 74.984 (133361/s), 70.837 (141169/s), 74.967 (133392/s),'
    pq_times[32] = 'pq encode lut float dist (10x5): 78.809 (126889/s), 79.34 (126039/s), 78.565 (127283/s), 79.171 (126308/s), 78.372 (127596/s), 78.689 (127082/s), 78.094 (128050/s), 80.031 (124951/s), 93.367 (107104/s), 81.896 (122106/s),'

    opq_times = {}
    opq_times[8] = 'opq encode lut float dist (10x5): 155.68 (64234/s), 159.49 (62698/s), 160.64 (62249/s), 158.21 (63205/s), 159.37 (62747/s), 159.29 (62778/s), 160.81 (62186/s), 158.5 (63090/s), 155.22 (64423/s), 158.98 (62901/s),'
    opq_times[16] = 'opq encode lut float dist (10x5): 170.42 (58677/s), 168.41 (59380/s), 169.12 (59129/s), 171.53 (58298/s), 167.32 (59766/s), 168.96 (59185/s), 170.43 (58676/s), 170.7 (58581/s), 169.86 (58870/s), 160.43 (62333/s),'
    opq_times[32] = 'opq encode lut float dist (10x5): 170.86 (58527/s), 175.79 (56885/s), 169.86 (58870/s), 180.3 (55464/s), 172.46 (57983/s), 171.66 (58254/s), 167.23 (59799/s), 168.19 (59457/s), 164.47 (60801/s), 168.31 (59413/s),'

    bolt_times = {}
    bolt_times[8] = 'bolt encode lut (10x5): 2.907 (3439972/s), 2.911 (3435245/s), 2.902 (3445899/s), 2.899 (3449465/s), 2.907 (3439972/s), 2.908 (3438789/s), 2.908 (3438789/s), 2.906 (3441156/s), 2.906 (3441156/s), 2.908 (3438789/s),'
    bolt_times[16] = 'bolt encode lut (10x5): 2.957 (3381805/s), 2.953 (3386386/s), 2.957 (3381805/s), 2.943 (3397893/s), 2.949 (3390979/s), 2.95 (3389830/s), 2.946 (3394433/s), 3.103 (3222687/s), 2.944 (3396739/s), 3.029 (3301419/s),'
    bolt_times[32] = 'bolt encode lut (10x5): 2.511 (3982477/s), 2.51 (3984063/s), 2.587 (3865481/s), 2.508 (3987240/s), 2.847 (3512469/s), 2.508 (3987240/s), 2.508 (3987240/s), 2.769 (3611412/s), 2.729 (3664345/s), 2.556 (3912363/s),'

    out_dicts = []
    algos = ['Bolt', 'PQ', 'OPQ']
    dicts = [bolt_times, pq_times, opq_times]
    for algo, d in zip(algos, dicts):
        for nbytes, s in list(d.items()):
            thruputs = _extract_thruput(s)
            out_dicts += [{'algo': algo, 'nbytes': nbytes, 'y': t} for t in thruputs]

    return pd.DataFrame.from_records(out_dicts)


def query_speed_results():
    # NOTE: all thruputs in this function (except matmul ones) need be
    # multiplied by 100,000 because we're reporting distances/sec, not time
    # to query 100k points

    bolt_times = {}
    bolt_times[8] = '4.385 (22805/s), 4.385 (22805/s), 4.408 (22686/s), 4.385 (22805/s), 5.117 (19542/s), 4.378 (22841/s), 4.392 (22768/s), 4.393 (22763/s), 4.381 (22825/s), 4.383 (22815/s)'
    bolt_times[16] = '8.268 (12094/s), 9.807 (10196/s), 8.389 (11920/s), 8.681 (11519/s), 8.711 (11479/s), 8.293 (12058/s), 9.797 (10207/s), 8.32 (12019/s), 9.767 (10238/s), 9.499 (10527/s)'
    bolt_times[32] = '19.385 (5158/s), 17.215 (5808/s), 18.612 (5372/s), 18.117 (5519/s), 17.323 (5772/s), 18.436 (5424/s), 18.979 (5268/s), 16.274 (6144/s), 19.696 (5077/s), 17.026 (5873/s)'

    popcnt_times = {}
    popcnt_times[8] = '2.456 (1302931596/s), 2.344 (1365187713/s), 2.125 (1505882352/s), 2.829 (1131141746/s), 2.148 (1489757914/s), 2.167 (1476695892/s), 2.327 (1375161151/s), 2.145 (1491841491/s), 2.12 (1509433962/s), 2.112 (1515151515/s)'
    popcnt_times[16] = '4.368 (732600732/s), 4.121 (776510555/s), 3.926 (815078960/s), 4.105 (779537149/s), 4.176 (766283524/s), 4.119 (776887594/s), 4.464 (716845878/s), 4.153 (770527329/s), 4.364 (733272227/s), 4.198 (762267746/s)'
    popcnt_times[32] = '7.612 (420388859/s), 7.347 (435551925/s), 7.694 (415908500/s), 9.122 (350800263/s), 7.343 (435789186/s), 9.344 (342465753/s), 8.148 (392734413/s), 9.046 (353747512/s), 8.455 (378474275/s), 7.685 (416395575/s)'

    pq_times = {}
    pq_times[8] = '36.499 (2739/s), 35.729 (2798/s), 36.521 (2738/s), 37.924 (2636/s), 37.079 (2696/s), 36.444 (2743/s), 36.115 (2768/s), 36.955 (2705/s), 35.913 (2784/s), 40.354 (2478/s)'
    pq_times[16] = '79.482 (1258/s), 82.546 (1211/s), 84.992 (1176/s), 84.996 (1176/s), 86.218 (1159/s), 84.495 (1183/s), 90.637 (1103/s), 82.164 (1217/s), 85.954 (1163/s), 82.255 (1215/s)'
    pq_times[32] = '214.85 (465/s), 217.41 (459/s), 212.49 (470/s), 210.75 (474/s), 211.12 (473/s), 212.54 (470/s), 209.91 (476/s), 219.95 (454/s), 212.97 (469/s), 213.44 (468/s)'

    opq_times = {}
    opq_times[8] = '38.653 (2587/s), 36.958 (2705/s), 37.684 (2653/s), 35.902 (2785/s), 38.032 (2629/s), 39.511 (2530/s), 42.321 (2362/s), 38.94 (2568/s), 39.224 (2549/s), 39.06 (2560/s)'
    opq_times[16] = '82.636 (1210/s), 82.401 (1213/s), 88.424 (1130/s), 86.649 (1154/s), 83.329 (1200/s), 82.719 (1208/s), 82.281 (1215/s), 80.581 (1240/s), 80.777 (1237/s), 81.107 (1232/s)'
    opq_times[32] = '221.61 (451/s), 230.01 (434/s), 241.68 (413/s), 222.39 (449/s), 215.13 (464/s), 215.49 (464/s), 212.27 (471/s), 213.95 (467/s), 213.96 (467/s), 217.79 (459/s)'

    # 1, 16 -> rowmajor times; 64, 256, 1024 -> colmajor times; (ie, use times from best layout)
    matmul1_times = '12.063 (8289811/s), 11.231 (8903926/s), 10.283 (9724788/s), 10.864 (9204712/s), 10.492 (9531071/s), 10.877 (9193711/s), 10.79 (9267840/s), 10.85 (9216589/s), 11.041 (9057150/s), 10.647 (9392317/s)'
    matmul16_times = '21.707 (73708941/s), 21.38 (74836295/s), 21.71 (73698756/s), 21.54 (74280408/s), 21.454 (74578167/s), 21.989 (72763654/s), 22.486 (71155385/s), 22.048 (72568940/s), 23.18 (69025021/s), 21.771 (73492260/s)'
    matmul64_times = '56.496 (113282356/s), 55.488 (115340253/s), 54.853 (116675478/s), 56.689 (112896681/s), 56.482 (113310435/s), 55.644 (115016893/s), 54.623 (117166761/s), 55.773 (114750865/s), 54.726 (116946241/s), 54.918 (116537383/s)'
    matmul256_times = '164.72 (155414306/s), 168.41 (152014488/s), 169.93 (150652927/s), 164.99 (155157157/s), 166.66 (153609831/s), 163.04 (157012830/s), 167.45 (152880544/s), 161.06 (158949936/s), 171.13 (149594750/s), 168.49 (151940505/s)'
    matmul1024_times = '653.63 (156664035/s), 677.26 (151197248/s), 692.88 (147788938/s), 664.79 (154032909/s), 702.61 (145742096/s), 651.74 (157116904/s), 656.4 (156003388/s), 664.69 (154056314/s), 665.34 (153906736/s), 651.88 (157083643/s)'

    out_dicts = []
    algos = ['Bolt', 'PQ', 'OPQ', 'Binary Embedding']
    dicts = [bolt_times, pq_times, opq_times, popcnt_times]

    for algo, d in zip(algos, dicts):
        for nbytes, s in list(d.items()):
            thruputs = _extract_thruput(s) * 1e5
            if algo == 'Binary Embedding':
                thruputs /= 1e5  # these are already dists/sec, not qps
            out_dicts += [{'algo': algo, 'nbytes': nbytes, 'y': t} for t in thruputs]

    matmul_strs = [matmul1_times, matmul16_times, matmul64_times, matmul256_times, matmul1024_times]
    batch_sizes = [1, 16, 64, 256, 1024]
    nbytes_list = [8, 16, 32]  # replicate results in each plot
    for s, sz in zip(matmul_strs, batch_sizes):
        algo = 'Matmul {}'.format(sz)
        for nbytes in nbytes_list:
            thruputs = _extract_thruput(s)
            out_dicts += [{'algo': algo, 'nbytes': nbytes, 'y': t} for t in thruputs]

    return pd.DataFrame.from_records(out_dicts)


def main():
    pass
    # print _extract_thruput('foo (10x5): 2.456 (1302931596/s), 2.344 (1365187713/s), 2.125 (1505882352/s), 2.829 (1131141746/s), 2.148 (1489757914/s), 2.167 (1476695892/s), 2.327 (1375161151/s), 2.145 (1491841491/s), 2.12 (1509433962/s), 2.112 (1515151515/s)')

    # print McqResults('../results/tmp.txt')
    # print McqResults('../results/mcq/mcq_D=256_M=8.txt')

    # res = query_speed_results()
    # print res.loc[res['algo'] == 'Matmul 1']
    # print res.loc[res['algo'] == 'Matmul 256']


if __name__ == '__main__':
    main()
