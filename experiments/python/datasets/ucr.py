#!/usr/bin/env/python

import os
import numpy as np
from joblib import Memory

from . import paths


_memory = Memory('.', verbose=1, compress=9)

UCR_DATASETS_DIR = paths.UCR


# ================================================================
# Public
# ================================================================

def all_ucr_datasets():
    for dataDir in sorted(all_ucr_dataset_dirs()):
        yield UCRDataset(dataDir)


class UCRDataset(object):

    def __init__(self, dataset_dir, sep='\t'):
        self.name = dataset_dir
        self.X_train, self.y_train = read_ucr_train_data(dataset_dir, sep=sep)
        self.X_test, self.y_test = read_ucr_test_data(dataset_dir, sep=sep)
        self.name = name_from_dir(dataset_dir)


def all_ucr_dataset_dirs():
    return _ucr_datasets_in_dir(UCR_DATASETS_DIR)


# ================================================================
# Private
# ================================================================

def _ucr_datasets_in_dir(dirpath):
    datasetsPath = os.path.expanduser(dirpath)
    files = os.listdir(datasetsPath)
    rm_dir = 'Missing_value_and_variable_length_datasets_adjusted'
    if rm_dir in files:
        files.remove(rm_dir)
    for i in range(len(files)):
        files[i] = os.path.join(datasetsPath, files[i])
    dirs = filter(os.path.isdir, files)
    return dirs


@_memory.cache
def _readtxt(path, sep=None):
    return np.genfromtxt(path, delimiter=sep)


def read_data_file(path, sep=None, mean_norm=False):
    D = _readtxt(path, sep=sep)
    labels = D[:, 0].astype(np.int)
    X = D[:, 1:]
    if mean_norm:
        X -= np.mean(X, axis=1, keepdims=True)
    return (X, labels)


def name_from_dir(datasetDir):
    return os.path.basename(datasetDir)


def dir_from_name(datasetName):
    return os.path.join(paths.UCR, datasetName)


def read_ucr_data_in_dir(datasetDir, train, sep=None):
    datasetName = name_from_dir(datasetDir)
    if train:
        fileName = datasetName + "_TRAIN.tsv"
    else:
        fileName = datasetName + "_TEST.tsv"
    filePath = os.path.join(datasetDir, fileName)
    return read_data_file(filePath, sep=sep)


def read_ucr_train_data(datasetDir, sep=None):
    return read_ucr_data_in_dir(datasetDir, train=True, sep=sep)


def read_ucr_test_data(datasetDir, sep=None):
    return read_ucr_data_in_dir(datasetDir, train=False, sep=sep)


# combines train and test data
def read_all_ucr_data(ucrDatasetDir):
    X_train, Y_train = read_ucr_train_data(ucrDatasetDir)
    X_test, Y_test = read_ucr_test_data(ucrDatasetDir)
    X = np.r_[X_train, X_test]
    Y = np.r_[Y_train, Y_test]
    return X, Y


# ================================================================ Main

@_memory.cache
def _load_ucr_stats_df():
    import pandas as pd
    stats = []
    for i, datasetDir in enumerate(all_ucr_dataset_dirs()):
        # Xtrain, _ = read_ucr_train_data(datasetDir)
        # Xtest, Ytest = read_ucr_test_data(datasetDir)
        dset = UCRDataset(datasetDir)

        N, D = dset.X_train.shape
        M, D = dset.X_test.shape
        nclasses = len(np.unique(dset.y_test))
        stats.append({'Dataset': dset.name, 'N': N, 'D': D, 'M': M,
                      'nclasses': nclasses})

        # print('%30s:\t%d\t%d\t%d\t%d' % (name_from_dir(datasetDir),
        #       N, M, D, nclasses)

    return pd.DataFrame.from_records(stats)


def main():
    # dsets = all_ucr_datasets()
    # for dset in dsets:
    #     print("loaded ucr dset:", dset.name)
    # # return

    df = _load_ucr_stats_df()

    # df = df.sort_values(axis=1)
    # df = df.loc[df['N'] > 100]
    df = df.loc[df['M'] > 100]
    print("ucr dset stats:")
    # print(df['M'].sort_values(ascending=False))
    print("number of dsets:", df.shape[0])
    print("mean, median Ntrain: ", df['N'].mean(), df['N'].median())
    print("mean, median Ntest: ", df['M'].mean(), df['M'].median())
    print("mean, median length: ", df['D'].mean(), df['D'].median())

    mvals = df['M'].to_numpy()
    mvals = np.sort(mvals)
    length = len(mvals)

    total_sizes = np.array([m * (length - i) for i, m in enumerate(mvals)])
    max_idx = np.argmax(total_sizes)
    best_m_cutoff = mvals[max_idx]
    print("best num dsets, m, sz = ",
          length - max_idx, best_m_cutoff, total_sizes[max_idx])

    print("type of mvals: ", type(mvals))
    for cutoff in [100, 200, 256, 300, 400, 500, 512, 1000]:
        ndsets = (mvals >= cutoff).sum()
        total_sz = total_sizes[ndsets-1]
        print(f"n >= {cutoff}: {ndsets} dsets, total sz = {total_sz}")

    # import matplotlib.pyplot as plt
    # xvals = length - np.arange(length)
    # # xvals = np.arange(length)
    # # plt.plot(xvals, total_sizes[::-1])
    # plt.plot(xvals, total_sizes)
    # plt.plot(xvals, mvals)
    # plt.show()

    df = df.loc[df['M'] >= best_m_cutoff]
    print("---- after cutting off M to maximize mat sizes:")
    print("mean, median Ntrain: ", df['N'].mean(), df['N'].median())
    print("mean, median Ntest: ", df['M'].mean(), df['M'].median())
    print("mean, median length: ", df['D'].mean(), df['D'].median())
    print("mean, median nclasses: ", df['nclasses'].mean(), df['nclasses'].median())
    print("min, max nclasses: ", df['nclasses'].min(), df['nclasses'].max())


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.3}".format(f)})
    main()
