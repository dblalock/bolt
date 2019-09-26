#!/bin/env/python

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import pathlib as pl

from . import files


RESULTS_DIR = pl.Path('results/amm')
FIGS_SAVE_DIR = pl.Path('../figs/amm')


if not os.path.exists(FIGS_SAVE_DIR):
    FIGS_SAVE_DIR.mkdir(parents=True)


def save_fig(name):
    plt.savefig(os.path.join(FIGS_SAVE_DIR, name + '.png'),
                dpi=300, bbox_inches='tight')


def make_cifar_fig(y_metric='Accuracy'):
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))

    df10 = pd.read_csv(RESULTS_DIR / 'cifar10.csv')
    df100 = pd.read_csv(RESULTS_DIR / 'cifar100.csv')
    dfs = (df10, df100)

    print("df10 len", len(df10))
    print("df100 len", len(df100))

    for df in dfs:
        # for Exact, set d = D
        D = 512
        mask = df['d'].isna()
        df.loc[mask, 'd'] = D
        # df.loc[mask] = df['D'].loc[mask]
        df.rename({'method': 'Method', 'acc_amm': 'Accuracy',
                   'r_sq': 'R-Squared'}, axis=1, inplace=True)
        df['d'] = np.log2(df['d']).astype(np.int32)

    # # print(df10)
    # print(df10['Method'].unique())
    # print(df10['d'].unique())

    def lineplot(data, ax):
        sb.lineplot(data=data, hue='Method', x='d', y=y_metric,
                    style='Method', markers=True, dashes=False, ax=ax)

    lineplot(df10, axes[0])
    lineplot(df100, axes[1])

    # plt.suptitle('Sketch size vs Classification Accuracy')
    plt.suptitle('Sketch size vs {}'.format(y_metric))
    axes[0].set_title('CIFAR-10')
    for ax in axes:
        ax.set_ylabel(y_metric)
    axes[1].set_xlabel('Log2(Sketch Size)')
    axes[1].set_title('CIFAR-100')
    plt.tight_layout()
    plt.subplots_adjust(top=.92, bottom=.2)
    save_fig('cifar_{}'.format(y_metric))


def main():
    make_cifar_fig('Accuracy')
    make_cifar_fig('R-Squared')
    print("done")


if __name__ == '__main__':
    main()
