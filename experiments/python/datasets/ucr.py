#!/usr/bin/env/python

import os
import numpy as np
from joblib import Memory
import pandas as pd

from . import paths


_memory = Memory('.', verbose=1, compress=9)

UCR_DATASETS_DIR = paths.UCR
UCR_INFO_PATH = paths.UCR_INFO


# ================================================================
# Public
# ================================================================

def all_ucr_datasets():
    for dataDir in sorted(all_ucr_dataset_dirs()):
        yield UCRDataset(dataDir)


class UCRDataset(object):

    def __init__(self, dataset_dir, sep='\t', precondition=True, znorm=True):
        self.name = name_from_dir(dataset_dir)
        self.X_train, y_train = read_ucr_train_data(dataset_dir, sep=sep)
        self.X_test, y_test = read_ucr_test_data(dataset_dir, sep=sep)

        # self.y_train = y_train
        # self.y_test = y_test

        all_lbls = np.r_[y_train, y_test]
        uniq_lbls = np.unique(all_lbls)
        new_lbls = np.argsort(uniq_lbls)  # same if labels are 0..(nclasses-1)
        mapping = dict(zip(uniq_lbls, new_lbls))
        self.y_train = np.array([mapping[lbl] for lbl in y_train])
        self.y_test = np.array([mapping[lbl] for lbl in y_test])

        # self.nclasses = len(uniq_lbls)

        # MelbournePedestrian has nans, even though not in missing data list
        for X in (self.X_train, self.X_test):
            for d in range(X.shape[1]):
                col = X[:, d]
                nan_idxs = np.isnan(col)
                if nan_idxs.sum() > 0:
                    # print("self.name: ", self.name)
                    # print("original number of nans: ", np.sum(nan_idxs))
                    # X[nan_idxs, d] = col.mean()
                    fillval = np.nanmedian(col)
                    if np.isnan(fillval):
                        # handle all-nan cols, which happens in Crop
                        fillval = np.nanmedian(X)
                    col[nan_idxs] = fillval
                    # np.nan_to_num(col, copy=False, nan=np.median(col))
                    # print("new number of nans: ", np.isnan(X[:, d]).sum())
                    # print("new number of nans: ", np.isnan(col).sum())

        if znorm:
            self.X_train -= self.X_train.mean(axis=1, keepdims=True)
            self.X_test -= self.X_test.mean(axis=1, keepdims=True)
            eps = 1e-20
            self.X_train *= 1 / (self.X_train.std(axis=1, keepdims=True) + eps)
            self.X_test *= 1 / (self.X_test.std(axis=1, keepdims=True) + eps)
        elif precondition:
            # weaker than znormalization since one offset and scale applied
            # to all dims and all samples in both train and test sets; this
            # is basically just here because the values in MelbournePedestrian
            # are huge and screw up numerical algorithms
            self.orig_mean = np.mean(self.X_train)
            self.X_train -= self.orig_mean
            self.X_test -= self.orig_mean
            self.orig_std = np.std(self.X_train)
            self.X_train /= self.orig_std
            self.X_test /= self.orig_std

        assert len(self.X_train) == len(self.y_train)
        assert len(self.X_test) == len(self.y_test)

        # if self.name == 'MelbournePedestrian':
        #     print("I am MelbournePedestrian!")
        #     print('new labels: ', new_lbls)
        #     print("X_train num nans", np.sum(np.isnan(self.X_train)))
        #     print("X_test num nans", np.sum(np.isnan(self.X_test)))
        #     # import sys; sys.exit()

        # if self.name == 'Wafer':
        #     print("original uniq labels train", np.unique(self.y_train))
        #     print("original uniq labels test", np.unique(self.y_test))


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
    dirs = list(filter(os.path.isdir, files))
    return dirs


@_memory.cache
def _readtxt(path, sep=None):
    return np.genfromtxt(path, delimiter=sep).astype(np.float32)


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


@_memory.cache
def load_ucr_dset_stats():
    df = pd.read_csv(UCR_INFO_PATH)
    df['l2-1nn-acc'] = 1. - df['ED (w=0)']
    return df


# ================================================================ Main

@_memory.cache
def _load_ucr_stats_df():

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
    # df = df.loc[df['M'] > 100]
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

    # df = df.loc[df['M'] >= best_m_cutoff]
    # print("---- after cutting off M to maximize mat sizes:")
    df = df.loc[df['N'] >= 128]
    print("---- after cutting off N to number of centroids:")
    print("number of dsets: ", len(df))
    print("mean, median Ntrain: ", df['N'].mean(), df['N'].median())
    print("mean, median Ntest: ", df['M'].mean(), df['M'].median())
    print("mean, median length: ", df['D'].mean(), df['D'].median())
    print("mean, median nclasses: ", df['nclasses'].mean(), df['nclasses'].median())
    print("min, max nclasses: ", df['nclasses'].min(), df['nclasses'].max())


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.3}".format(f)})
    main()
