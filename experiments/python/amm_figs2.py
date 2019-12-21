#!/bin/env/python

import collections
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import pathlib as pl

# from . import files
from . import amm_results2 as res
# from . import amm_methods as ameth

# sb.set_context('poster')
# sb.set_context('talk')
# sb.set_cmap('tab10')

FIGS_SAVE_DIR = pl.Path('../figs/amm')
USE_FONT = 'DejaVu Sans'


if not os.path.exists(FIGS_SAVE_DIR):
    FIGS_SAVE_DIR.mkdir(parents=True)


def save_fig(name):
    plt.savefig(os.path.join(FIGS_SAVE_DIR, name + '.png'),
                dpi=300, bbox_inches='tight')


def rename_values_in_col(df, col, name_map, drop_others=True):
    vals = [name_map.get(name.strip().lower(), "") for name in df[col]]
    valid_vals = set(name_map.values())
    # print("valid_vals: ", valid_vals)
    valid_mask = np.array([val in valid_vals for val in vals])
    # print("valid mask: ", valid_mask)
    df[col] = vals
    if drop_others:
        df = df.loc[valid_mask]
    return df
    # print(df)


def melt_observation_cols(df, cols, var_name=None, value_name=None):
    """like pd.melt, but assumes only 1 observation var instead of 1 id var"""
    independent_vars = [col for col in df.columns
                        if col not in set(cols)]
    var_name = 'trial'
    return pd.melt(df, id_vars=independent_vars, value_vars=cols,
                   var_name=var_name, value_name='time')


def melt_times(df, ntimes=5):
    observation_vars = 't0 t1 t2 t3 t4'.split()
    observation_vars = observation_vars[:ntimes]
    return melt_observation_cols(
        df, observation_vars, var_name='trial', value_name='time')


def add_ylabels_on_right(axes, fmt, vals):
    for i, ax in enumerate(axes):
        lbl = fmt.format(vals[i])
        ax2 = ax.twinx()
        ax2.get_xaxis().set_visible(False)
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel(lbl, fontsize=14, family=USE_FONT)
        sb.despine(ax=ax2, top=True, left=True, bottom=True, right=True)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_yticks([])
        ax2.tick_params(axis='y', which='y', length=0)


def scan_speed_fig(save=True):

    # ================================ data cleaning

    df = res.scan_timings()

    name_map = collections.OrderedDict()
    name_map['mithral scan'] = 'Mithral'
    # name_map['bolt scan uint8'] = 'Bolt\nCheating'
    name_map['bolt scan safe uint16'] = 'Bolt'
    name_map['popcount scan'] = 'Popcount'
    name_map['pq scan'] = 'PQ / OPQ'
    df = rename_values_in_col(df, 'algo', name_map)
    df = melt_times(df)

    # alright, can't get stds to show without really screwing with stuff
    # times = np.array(df['time'])
    # times += np.random.randn(len(df['time'])) * .1  # get 1px for stds
    # # mask = df['algo'] == 'PQ / OPQ'
    # mask = df['B'] == 64
    # df['time'].loc[mask] = times[mask]

    df['thruput'] = df['N'] * df['M'] / df['time']
    df['thruput'] /= 1e6  # just use units of billions; times are in ms
    # df['thruput'] *= 1e3  # times are in ms

    # ================================ fig creation

    sb.set_context("talk")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    axes = [ax]

    sb.barplot(data=df, x='algo', y='thruput', units='trial',
               hue='B', hue_order=[8, 16, 32, 64], order=name_map.values(),
               ax=ax, ci='sd')

    # ------------------------ clean up / format axes

    for ax in axes[:-1]:
        # remove x labels except for bottom axis
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.get_xaxis().set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    labels = ['8B Codes', '16B Codes', '32B Codes', '64B Codes']
    # labels = ['8 Bytes', '16 Bytes', '32 Bytes', '64 Bytes']
    # labels = ['8B', '16B', '32B', '64B']
    plt.figlegend(handles, labels, loc='lower center', ncol=4, fontsize=14)

    for ax in axes:
        ax.set_ylabel('Billion Dot Products/s', family=USE_FONT)
        ax.get_legend().remove()

        # ------------------------ have bottom / top axes print title, x info

    axes[0].set_title('Speed of f() Functions for Different Encoding Sizes',
                      y=1.04, family=USE_FONT, fontsize=18)

    # get and set them again so we can make the first one bold; can't make
    # it bold beforehand because need a tick lbl object, not a string
    xlabels = list(axes[-1].get_xticklabels())
    xlabels[0].set_weight('bold')
    # axes[-1].set_xticklabels(xlabels, rotation=60, ha='right')
    axes[-1].set_xticklabels(xlabels)

    axes[-1].tick_params(axis='x', which='major', pad=4)
    axes[-1].set_xlabel("", labelpad=-30)

    ax.xaxis.set_ticks_position('none')

    # ------------------------ save / show plot

    plt.tight_layout()
    plt.subplots_adjust(bottom=.21)

    if save:
        save_fig('scan_speed')
    else:
        plt.show()


def encode_speed_fig(save=True):
    # ================================ data cleaning
    df = res.encode_timings()

    name_map = collections.OrderedDict()
    name_map['mithral encode 4b i8'] = 'Mithral i8'
    name_map['mithral encode 4b i16'] = 'Mithral i16'
    name_map['mithral encode 4b f32'] = 'Mithral f32'
    name_map['bolt encode'] = 'Bolt'
    name_map['pq encode'] = 'PQ'
    name_map['opq encode'] = 'OPQ'
    df = rename_values_in_col(df, 'algo', name_map)
    df = melt_times(df, ntimes=3)  # TODO rerun with ntrials=5

    df['thruput'] = df['N'] * df['D'] / df['time']
    # df['thruput'] /= 1e6  # just use units of billions; times are in ms
    df['B'] = df['C'] / 2

    # ================================ fig creation

    sb.set_context('talk')
    # sb.set_style('darkgrid')
    sb.set_style('white')

    # use_nbytes = [8, 16, 32, 64]
    use_nbytes = [8, 16, 32]

    fig, axes = plt.subplots(len(use_nbytes), 1, figsize=(6, 8), sharey=True)
    for i, nbytes in enumerate(use_nbytes):
        data = df.loc[df['B'] == nbytes]
        sb.lineplot(data=data, x='D', y='thruput', hue='algo', units='trial',
                    ax=axes[i], ci='sd', estimator=None)

    # ------------------------ axis cleanup
    axes[0].set_title('Speed of g() Functions\nfor Different Encoding Sizes',
                      y=1.04, family=USE_FONT, fontsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]  # rm df column name
    plt.figlegend(handles, labels, loc='lower center', ncol=3, fontsize=13)

    for ax in axes:
        ax.semilogx()
        ax.semilogy()
        ax.get_legend().remove()
        # ax.set_ylabel('Billions of\nScalars Encoded/s',
        # ax.set_ylabel('Scalars Encoded/s\n(Billions)',
        # ax.set_ylabel('Scalars Encoded\nper Second (Billions)',
        # ax.set_ylabel('Scalars Encoded\nper Second',
        ax.set_ylabel('Scalars Encoded/s',
                      family=USE_FONT, fontsize=14)
    for ax in axes[:-1]:
        # remove x labels except for bottom axis
        ax.tick_params(axis='x', which='x', length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("", visible=False)
        # ax.get_xaxis().set_visible(False)
        # ax.get_xticklabels().set_visible(False)
    axes[-1].set_xlabel('Number of Columns in Matrix A',
                        family=USE_FONT, fontsize=14)

    # add byte counts on the right
    add_ylabels_on_right(axes, "{}B Encodings", use_nbytes)

    plt.tight_layout()
    plt.subplots_adjust(bottom=.18, hspace=.15)

    if save:
        save_fig('encode_speed')
    else:
        plt.show()


def lut_speed_fig(save=True):
    # ================================ data cleaning
    df = res.lut_timings()

    name_map = collections.OrderedDict()
    name_map['mithral lut dense'] = 'Mithral'
    name_map['mithral lut sparse'] = 'Mithral'
    name_map['bolt lut'] = 'Bolt'
    name_map['pq lut'] = 'PQ'
    name_map['opq lut'] = 'OPQ'
    df = rename_values_in_col(df, 'algo', name_map)
    # print(df[:20])

    # df['lutconst'] = df['lutconst'].str.strip().astype(np.float).astype(np.int)
    # print("df.dtypes", df.dtypes)
    # import sys; sys.exit()

    names = list(df['algo'])
    consts = np.array(df['lutconst'])
    # print("len(names)", len(names))
    # print("len(consts)", len(consts))

    mithral_const_to_name = collections.OrderedDict()
    mithral_const_to_name[-1] = 'Mithral, L = ∞'
    mithral_const_to_name[4] = 'Mithral, L = 4'
    mithral_const_to_name[2] = 'Mithral, L = 2'
    mithral_names = list(mithral_const_to_name.values())

    # add lut constant into the name for mithral variations
    new_names = []
    ismithral = []
    for i, name in enumerate(names):
        if not name.startswith('Mithral'):
            new_names.append(name)
            ismithral.append(False)
            continue
        # const = consts[i]
        # const = "{:d}".format(int(const)) if const > 0 else "∞"
        # new_names.append(f"{name}, L = {const}")
        new_names.append(mithral_const_to_name[int(consts[i])])
        ismithral.append(True)
    # print("len(new_names)", len(new_names))
    df['algo'] = new_names
    df['ismithral'] = ismithral

    df = melt_times(df, ntimes=5)
    # df = melt_times(df, ntimes=3)  # TODO rerun with ntrials=5
    # print(df)

    df['thruput'] = df['N'] * df['D'] / df['time']
    # df['thruput'] /= 1e6  # just use units of billions; times are in ms

    # # TODO rm once we have updated results
    # mask = df['algo'].isin(('PQ', 'OPQ'))
    # df['B'] = -1  # create placeholder col
    # df['B'].loc[mask] = df['C'].loc[mask]
    # df['B'].loc[~mask] = df['C'].loc[~mask] / 2

    # ================================ fig creation

    sb.set_context('talk')
    # sb.set_style('darkgrid')
    sb.set_style('white')

    # use_nbytes = [8, 16, 32, 64]
    use_nbytes = [8, 16, 32]

    fig, axes = plt.subplots(len(use_nbytes), 1, figsize=(6, 8), sharey=True)
    order = [mithral_names[2], 'Bolt',
             mithral_names[1], 'PQ',
             mithral_names[0], 'OPQ']
    dashes = {k: ('-' if k in mithral_names else '--') for k in order}
    # dashes = {k: ('solid' if k in mithral_names else 'dashed') for k in order}
    # dashes = {k: (None if k in mithral_names else [3, 3]) for k in order}
    # dashes = True
    # print(dashes)
    # import sys; sys.exit()
    for i, nbytes in enumerate(use_nbytes):
        data = df.loc[df['B'] == nbytes]
        ax = axes[i]
        print(f"------------------------ {nbytes}B")
        for algo in order:
            subdf = data.loc[df['algo'] == algo]
            print("plotting algo: ", algo)
            x = subdf['D'].as_matrix()
            y = subdf['thruput'].as_matrix()
            sort_idxs = np.argsort(x)
            x, y = x[sort_idxs], y[sort_idxs]
            ax.plot(x, y, dashes[algo], label=algo)
        # sb.lineplot(data=data, x='D', y='thruput', hue='algo', units='trial',
        #             hue_order=order,
        #             # hue_order=order, style='algo', style_order=order,
        #             # dashes=True,
        #             style='ismithral', style_order=[True, False], dashes=True,
        #             ax=axes[i], ci='sd', estimator=None)

    # ------------------------ axis cleanup
    axes[0].set_title('Speed of h() Functions\nfor Different Encoding Sizes',
                      y=1.04, family=USE_FONT, fontsize=18)

    # for ax in axes:
    #     print("ax handles, labels: ")
    #     print(ax.get_legend_handles_labels())

    handles, labels = axes[-1].get_legend_handles_labels()
    # handles, labels = handles[1:], labels[1:]  # rm df column name
    # handles, labels = handles[:-3], labels[:-3]  # rm ismithral
    plt.figlegend(handles, labels, loc='lower center', ncol=3, fontsize=13)

    for ax in axes:
        ax.semilogx()
        ax.semilogy()
        # ax.get_legend().remove()
        ax.set_ylabel('Scalars Encoded/s',
                      family=USE_FONT, fontsize=14)
    for ax in axes[:-1]:
        # remove x labels except for bottom axis
        ax.tick_params(axis='x', which='x', length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("", visible=False)
    axes[-1].set_xlabel('Number of Rows in Matrix B',
                        family=USE_FONT, fontsize=14)

    # add byte counts on the right
    add_ylabels_on_right(axes, "{}B Encodings", use_nbytes)

    plt.tight_layout()
    plt.subplots_adjust(bottom=.18, hspace=.15)

    if save:
        save_fig('lut_speed')
    else:
        plt.show()


def main():
    # scan_speed_fig()
    # encode_speed_fig()
    lut_speed_fig()


if __name__ == '__main__':
    main()
