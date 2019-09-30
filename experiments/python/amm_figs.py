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


def _xlabel_for_xmetric(x_metric):
    if x_metric == 'd':
        return 'Log2(Sketch Size)'
    elif x_metric == 'secs':
        return 'Time (s)'
    elif x_metric == 'muls':
        return 'Log10(# of Multiplies)'


def make_cifar_fig(x_metric='d', y_metric='Accuracy'):
    fig, axes = plt.subplots(2, 1, figsize=(6, 9), sharex=True)

    df10 = pd.read_csv(RESULTS_DIR / 'cifar10.csv')
    df100 = pd.read_csv(RESULTS_DIR / 'cifar100.csv')
    dfs = (df10, df100)

    # print("df10 len", len(df10))
    # print("df100 len", len(df100))

    for df in dfs:
        # for Exact, set d = D
        D = 512
        mask = df['d'].isna()
        df.loc[mask, 'd'] = D
        # df.loc[mask] = df['D'].loc[mask]
        df.rename({'method': 'Method', 'acc_amm': 'Accuracy',
                   'r_sq': 'R-Squared', 'nmultiplies': 'muls'},
                  axis=1, inplace=True)
        df['d'] = np.log2(df['d']).astype(np.int32)
        df['Log10(MSE)'] = np.log10(1. - df['R-Squared'] + 1e-10)
        # if 'nlookups' in df:
        df['muls'] = df['muls'].fillna(0)
        mask = ~df['nlookups'].isna()
        df['muls'].loc[mask] += df['nlookups'].loc[mask]
        # df['muls'].loc[mask].add(df['nlookups'].loc[mask], fill_value=0)
        # df['muls'].add(df['nlookups'], fill_value=0)
        # print("df muls, nlookups")
        # print(df[['muls', 'nlookups']])
        df['muls'] = np.log10(df['muls'])

    # print('opq results:')
    # df = dfs[0]
    # print(df.loc[df['Method'].isin(['OPQ', 'Exact']), ['Method', 'nmuls', 'muls', 'R-Squared']])

    # import sys; sys.exit()

    def lineplot(data, ax):
        sb.lineplot(data=data, hue='Method', x=x_metric, y=y_metric,
                    style='Method', markers=True, dashes=False, ax=ax)

    lineplot(df10, axes[0])
    lineplot(df100, axes[1])

    # plt.suptitle('Sketch size vs Classification Accuracy')
    xlbl = _xlabel_for_xmetric(x_metric)
    plt.suptitle('{} vs {}'.format(xlbl, y_metric))
    axes[0].set_title('CIFAR-10')
    for ax in axes:
        ax.set_ylabel(y_metric)
    axes[0].set_xlabel(None)
    axes[1].set_xlabel(xlbl)
    axes[1].set_title('CIFAR-100')
    plt.tight_layout()
    plt.subplots_adjust(top=.92, bottom=.2)
    save_fig('cifar_{}_{}'.format(x_metric, y_metric))


# def make_ecg_fig(y_metric='R-Squared'):
def make_ecg_fig(x_metric='d'):
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))
    df = pd.read_csv(RESULTS_DIR / 'ecg.csv')

    D = 24
    mask = df['d'].isna()
    df.loc[mask, 'd'] = D
    df.rename({'method': 'Method', 'acc_amm': 'Accuracy',
               'r_sq': 'R-Squared', 'nmultiplies': 'muls'},
              axis=1, inplace=True)
    df['d'] = np.log2(df['d'])
    df['Log10(MSE)'] = np.log10(1. - df['R-Squared'] + 1e-10)  # avoid log10(0)
    df['muls'] = df['muls'].fillna(0)
    mask = ~df['nlookups'].isna()
    df['muls'].loc[mask] += df['nlookups'].loc[mask]
    df['muls'] = np.log10(df['muls'])

    df['Compression Ratio'] = df['nbytes_orig'] / df['nbytes_blosc_byteshuf']

    def lineplot(data, ycol, ax):
        sb.lineplot(data=data, hue='Method', x=x_metric, y=ycol,
                    style='Method', markers=True, dashes=False, ax=ax)

    lineplot(df, ycol='R-Squared', ax=axes[0])
    lineplot(df, ycol='Compression Ratio', ax=axes[1])

    xlbl = _xlabel_for_xmetric(x_metric)
    axes[0].set_title('ECG: {} vs R-Squared'.format(xlbl))
    axes[1].set_title('ECG: {} vs Compression Ratio'.format(xlbl))
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('R-Squared')
    axes[1].set_ylabel('Compression Ratio')
    axes[1].set_xlabel(xlbl)

    plt.tight_layout()
    plt.subplots_adjust(top=.92, bottom=.2)
    save_fig('ecg_{}'.format(x_metric))


def make_caltech_fig(x_metric='d'):
    """x_metric should be in {'d', 'secs', 'muls'}"""

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    df = pd.read_csv(RESULTS_DIR / 'caltech.csv')

    # TODO factor out this block into a shared function
    mask = df['d'].isna()
    df.loc[mask, 'd'] = df.loc[mask, 'D']
    df.rename({'method': 'Method', 'r_sq': 'R-Squared', 'nmultiplies': 'muls'},
              axis=1, inplace=True)
    df['d'] = np.log2(df['d'])
    df['Log10(MSE)'] = np.log10(1. - df['R-Squared'] + 1e-10)  # avoid log10(0)
    df['muls'] = df['muls'].fillna(0)
    mask = ~df['nlookups'].isna()
    df['muls'].loc[mask] += df['nlookups'].loc[mask]
    df['muls'] = np.log10(df['muls'])

    # print("caltech df")
    # print(df[['Method', 'd', 'R-Squared', 'task_id']])

    # if x_metric == 'muls':
    #     df['muls'] =
    #     df_exact = df.loc[df['Method'] == 'Exact']
    #     for task_id in df_exact['task_id'].unique():
    #         subdf = df_exact.loc[df['task_id'] == task_id]

    # sb.lineplot(data=df, hue='Method', x='d', y='R-Squared',
    # sb.lineplot(data=df, hue='Method', x=x_metric, y='R-Squared',
    sb.lineplot(data=df, hue='Method', x=x_metric, y='Log10(MSE)',
                style='Method', markers=True, dashes=False, ax=ax)

    # ax.set_ylim([ax.get_ylim()[0], 1])
    # ax.set_ylim([0, 1])
    ax.set_ylabel('Log10(MSE + 1e-10)')
    if x_metric == 'd':
        ax.set_title('Caltech: Sketch Size vs Log Squared Error')
        ax.set_xlabel('Log2(Sketch Size)')
    elif x_metric == 'secs':
        ax.set_title('Caltech: Computation Time vs Log Squared Error')
        ax.set_xlabel('Time (s)')
    elif x_metric == 'muls':
        ax.set_title('Caltech: # of Multiplies vs Log Squared Error')
        ax.set_xlabel('Log10(# of Multiplies)')
    plt.tight_layout()
    plt.subplots_adjust(top=.92, bottom=.2)
    save_fig('caltech_{}'.format(x_metric))


def main():
    # for x_metric in 'd secs muls'.split():
    for x_metric in ['muls']:
        for y_metric in ('Accuracy', 'R-Squared'):
            make_cifar_fig(x_metric, y_metric)
    # make_cifar_fig('d', 'Accuracy')
    # make_cifar_fig('Accuracy')
    # make_cifar_fig('Accuracy')
    # make_cifar_fig('R-Squared')
    # make_ecg_fig(x_metric='d')
    # make_ecg_fig(x_metric='secs')
    # make_ecg_fig(x_metric='muls')
    # make_caltech_fig(x_metric='d')
    # make_caltech_fig(x_metric='secs')
    # make_caltech_fig(x_metric='muls')
    print("done")


if __name__ == '__main__':
    main()
