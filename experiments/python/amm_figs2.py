#!/bin/env/python

import collections
import os
import numpy as np
import matplotlib as mpl
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
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [USE_FONT]

# to avoid type3 fonts; 42 = truetype, which is more flexible than type1
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def fix_ticks():
    # recover from seaborn white style messing this up
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True


if not os.path.exists(FIGS_SAVE_DIR):
    FIGS_SAVE_DIR.mkdir(parents=True)


def set_seaborn_style(stylename):
    sb.set_style(stylename)
    fix_ticks()


def save_fig(name):
    # plt.savefig(os.path.join(FIGS_SAVE_DIR, name + '.png'),
    #             dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGS_SAVE_DIR, name + '.pdf'),
                bbox_inches='tight')


def _xlabel_for_xmetric(x_metric):
    return {'d': 'Sketch Size',
            'secs': 'Time (s)',
            'muls': 'Number of Multiplies',
            'nlookups': 'Number of Lookups',
            'ops': 'Number of Operations',
            'Latency': 'Latency (ms)',
            'Speedup': 'Speedup Over Exact Matrix Multiply',
            'NormalizedTime': 'Normalized Latency',
            'Throughput': 'Throughput (elements/s)'}[x_metric]


def _ylabel_for_xmetric(y_metric):
    if y_metric == 'Relative Accuracy':
        return 'Normalized\nAccuracy'
    if y_metric == 'Accuracy':
        return 'Classification\nAccuracy'
    return y_metric


def add_ylabels_on_right(axes, fmt, vals):
    for i, ax in enumerate(axes):
        lbl = fmt.format(vals[i])
        ax2 = ax.twinx()
        ax2.get_xaxis().set_visible(False)
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel(lbl, fontsize=14, family=USE_FONT, labelpad=5)
        sb.despine(ax=ax2, top=True, left=True, bottom=True, right=True)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_yticks([])
        ax2.tick_params(axis='y', which='y', length=0)


def scan_speed_fig(save=True):

    # ================================ data cleaning

    df = res.scan_timings()

    name_map = collections.OrderedDict()
    # name_map['mithral scan'] = 'Mithral'
    name_map['mithral scan'] = 'MADDNESS'
    # name_map['mithral scan'] = 'Maddness'
    # name_map['bolt scan uint8'] = 'Bolt\nCheating'
    name_map['bolt scan safe uint16'] = 'Bolt'
    name_map['popcount scan'] = 'Popcount'
    name_map['pq scan'] = 'PQ / OPQ'
    df = res.rename_values_in_col(df, 'algo', name_map)
    df = res.melt_times(df)

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
    # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    axes = [ax]

    sb.barplot(data=df, x='algo', y='thruput', units='timing_trial',
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
                      y=1.04, family=USE_FONT, fontsize=20)

    # # get and set them again so we can make the first one bold; can't make
    # # it bold beforehand because need a tick lbl object, not a string
    # xlabels = list(axes[-1].get_xticklabels())
    # xlabels[0].set_weight('bold')
    # # axes[-1].set_xticklabels(xlabels, rotation=60, ha='right')
    # axes[-1].set_xticklabels(xlabels)

    axes[-1].tick_params(axis='x', which='major', pad=4)
    axes[-1].set_xlabel("", labelpad=-30)

    ax.xaxis.set_ticks_position('none')

    # ------------------------ save / show plot

    plt.tight_layout()
    # plt.subplots_adjust(bottom=.21)
    plt.subplots_adjust(bottom=.23)

    if save:
        save_fig('scan_speed')
    else:
        plt.show()


def encode_speed_fig(save=True):
    # ================================ data cleaning
    df = res.encode_timings()

    df = df.loc[df['algo'] != 'mithral encode i16']

    # print("df ours f32: ", df.loc[df['algo'].str.lower().str.strip() == 'mithral encode f32'])
    # print("df ours f32: ", df.loc[df['algo'].str.lower().str.strip() == 'mithral encode i8'])

    # print(df)
    # # # print(df['B'])
    # # # print(df['C'])
    # import sys; sys.exit()

    name_map = collections.OrderedDict()
    # name_map['mithral encode i8'] = r'$\bf{Mithral}$ $\bf{i8}$')
    # name_map['mithral encode i8'] = r'$\bf{Mithral}$ $\bf{i8}$')
    # name_map['mithral encode i8'] = 'Mithral i8'
    # name_map['mithral encode i16'] = 'Mithral i16'  # no i16 in plot
    # name_map['mithral encode f32'] = 'Mithral f32'
    # name_map['mithral encode i8'] = 'MADDNESS i8'
    # name_map['mithral encode f32'] = 'MADDNESS f32'
    name_map['mithral encode f32'] = 'MADDNESS'
    name_map['bolt encode'] = 'Bolt'
    name_map['pq encode'] = 'PQ'
    name_map['opq encode'] = 'OPQ'
    df = res.rename_values_in_col(df, 'algo', name_map)
    df = res.melt_times(df, ntimes=5)
    order = 'MADDNESS Bolt PQ OPQ'.split()

    # df['thruput'] = df['N'] * df['D'] / df['time']
    # df['thruput'] = df['N'] / (df['time'] * .001)  # rows/sec

    time_secs = (df['time'] * .001)
    df['elemsz'] = 4
    df['elemsz'].loc[df['algo'].str.endswith('i8')] = 1
    df['elemsz'].loc[df['algo'].str.endswith('i16')] = 2
    df['thruput'] = df['N'] * df['D'] * df['elemsz'] / time_secs  # GB/sec
    df['thruput'] /= 1e9  # convert to GB

    # df['thruput'] /= 1e6  # just use units of billions; times are in ms
    # full_byte_per_codebook = df['algo'].isin(['PQ', 'OPQ'])
    # df['B'] = df['C'].values / 2

    # # cvals = df['C'].loc[full_byte_per_codebook]
    # df['B'].loc[full_byte_per_codebook] = df['C'].loc[full_byte_per_codebook]
    # df['B'] = df['B'].astype(np.int)

    # # print("df.cols: ", df.columns)
    # print(df)
    # # # print(df['B'])
    # # # print(df['C'])
    # import sys; sys.exit()

    # ================================ fig creation

    sb.set_context('talk')
    # sb.set_style('darkgrid')
    # sb.set_style('white')
    set_seaborn_style('white')

    # use_nbytes = [8, 16, 32, 64]
    use_nbytes = [8, 16, 32]

    # fig, axes = plt.subplots(len(use_nbytes), 1, figsize=(6, 8), sharey=True)
    # fig, axes = plt.subplots(len(use_nbytes), 1, figsize=(6, 6.5), sharey=True)
    # fig, axes = plt.subplots(len(use_nbytes), 1, figsize=(6, 7), sharey=True)
    fig, axes = plt.subplots(len(use_nbytes), 1, figsize=(6, 6.5), sharey=True)
    for i, nbytes in enumerate(use_nbytes):
        data = df.loc[df['B'] == nbytes]

        # print("df.cols: ", df.columns)
        # print(data)
        # # # print(df['B'])
        # # # print(df['C'])
        # import sys; sys.exit()

        order = name_map.values()
        dashes = {name: ([] if name.lower().startswith('maddness') else
                         mpl.rcParams['lines.dashed_pattern'])
                  for name in order}
        # dashes = None
        # sb.lineplot(data=data, x='D', y='thruput', hue='algo',
        # sb.lineplot(data=data, x='D', y='thruput', hue='algo', units='timing_trial',
        sb.lineplot(data=data, x='D', y='thruput', hue='algo',
                    # ax=axes[i], ci='sd', estimator=None, hue_order=order,
                    ax=axes[i], ci='sd', estimator='mean', hue_order=order,
                    # ax=axes[i], ci=None, estimator='mean', hue_order=order,
                    style='algo', style_order=order, dashes=dashes,
                    palette=my_colors_list)

        # import sys; sys.exit()

    # ------------------------ axis cleanup
    axes[0].set_title('Speed of g() Functions\nfor Different Encoding Sizes',
                      y=1.04, family=USE_FONT, fontsize=16)

    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]  # rm df column name
    # plt.figlegend(handles, labels, loc='lower center', ncol=3, fontsize=13)
    plt.figlegend(handles, labels, loc='lower center', ncol=4, fontsize=13)

    for ax in axes:
        # ax.semilogx()
        ax.semilogy()
        ax.set_ylim([.02, 1000])
        # ax.set_yticks([.1, 1, 10, 100, 1000])
        ax.set_yticks([.1, 10, 1000])
        ax.get_legend().remove()
        # ax.set_ylabel('Billions of\nScalars Encoded/s',
        # ax.set_ylabel('Scalars Encoded/s\n(Billions)',
        # ax.set_ylabel('Scalars Encoded\nper Second (Billions)',
        # ax.set_ylabel('Scalars Encoded\nper Second',
        # ax.set_ylabel('Scalars Encoded/s',
        # ax.set_ylabel('Rows Encoded/s',
        ax.set_ylabel('Encoding\nSpeed (GB/s)',
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
    # plt.subplots_adjust(bottom=.18, hspace=.15)
    # plt.subplots_adjust(bottom=.19, hspace=.15)
    plt.subplots_adjust(bottom=.17, hspace=.15)
    # plt.subplots_adjust(bottom=.21, hspace=.15)

    if save:
        save_fig('encode_speed')
    else:
        plt.show()


def lut_speed_fig(save=True):
    # ================================ data cleaning
    df = res.lut_timings()

    name_map = collections.OrderedDict()
    # name_map['mithral lut dense'] = '$\bf{Mithral}$'
    # name_map['mithral lut sparse'] = '$\bf{Mithral}$'
    name_map['mithral lut dense'] = 'MADDNESS'
    name_map['mithral lut sparse'] = 'MADDNESS'
    name_map['bolt lut'] = 'Bolt'
    name_map['pq lut'] = 'PQ'
    name_map['opq lut'] = 'OPQ'
    df = res.rename_values_in_col(df, 'algo', name_map)
    # print(df[:20])

    # df['lutconst'] = df['lutconst'].str.strip().astype(np.float).astype(np.int)
    # print("df.dtypes", df.dtypes)
    # import sys; sys.exit()

    names = list(df['algo'])
    consts = np.array(df['lutconst'])
    # print("len(names)", len(names))
    # print("len(consts)", len(consts))

    mithral_const_to_name = collections.OrderedDict()
    mithral_const_to_name[-1] = 'MADDNESS, L = ∞'
    mithral_const_to_name[4] = 'MADDNESS, L = 4'
    mithral_const_to_name[2] = 'MADDNESS, L = 2'
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

    df = res.melt_times(df, ntimes=5)
    # df = res.melt_times(df, ntimes=3)  # TODO rerun with ntrials=5
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
    # sb.set_style('white')
    set_seaborn_style('white')

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
        # print(f"------------------------ {nbytes}B")
        # manual version
        # for algo in order:
        #     subdf = data.loc[df['algo'] == algo]
        #     print("plotting algo: ", algo)
        #     x = subdf['D'].as_matrix()
        #     y = subdf['thruput'].as_matrix()
        #     sort_idxs = np.argsort(x)
        #     x, y = x[sort_idxs], y[sort_idxs]
        #     ax.plot(x, y, dashes[algo], label=algo)

        dashes = {name: ([] if name.lower().startswith('mithral') else
                         mpl.rcParams['lines.dashed_pattern'])
                  for name in order}
        sb.lineplot(data=data, x='D', y='thruput', hue='algo',
                    units='timing_trial', ax=axes[i], ci='sd',
                    estimator=None, hue_order=order,
                    style='algo', style_order=order, dashes=dashes)

        # sb.lineplot(data=data, x='D', y='thruput', hue='algo', units='timing_trial',
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
    handles, labels = handles[1:], labels[1:]  # rm df column name
    # handles, labels = handles[:-3], labels[:-3]  # rm ismithral
    plt.figlegend(handles, labels, loc='lower center', ncol=3, fontsize=13)

    for ax in axes:
        # ax.semilogx()
        ax.semilogy()
        ax.get_legend().remove()
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


def lotsa_colors_cmap(value):
    assert 0 <= value <= 1  # if this throws, I don't understand cmaps
    if value < .3333:
        return plt.get_cmap('tab20')(3 * value)
    elif value < .6666:
        return plt.get_cmap('tab20b')((3 * value) - 1)
    else:
        return plt.get_cmap('tab20c')((3 * value) - 2)


# def my_tab10(value):
#     assert 0 <= value <= 1
#     value = int(value * 10)
#     perm = [3, 1, 2, 4, 5, 6, 7, 8, 9]  # make red first, then orange
#     value = perm[value]
#     return plt.get_cmap('tab10')((value / 10.) + .01)

# def my_cmap(value):

my_colors_list = (plt.get_cmap('Set1').colors
                  + plt.get_cmap('Set3').colors[:1]  # skip light yellow
                  + plt.get_cmap('Set3').colors[2:]
                  + plt.get_cmap('Dark2').colors[:6])
# my_colors_list = my_colors_list[:5] + () my_colors_list[6:]  # rm bright yellow
# new_yellow = (240./255, 230./255, 140./255)
new_yellow = (204. / 255, 204. / 255, 0. / 255)
# print(type(my_colors_list))
# print(my_colors_list)
my_colors_list = my_colors_list[:5] + (new_yellow,) + my_colors_list[6:]
# print(type(my_colors_list))
# print(my_colors_list)

# import sys; sys.exit()

# DEFAULT_PLOT_METHODS = ('Mithral', 'MithralPQ', 'Brute Force', 'Bolt',
# DEFAULT_PLOT_METHODS = ('MADDNESS', 'MADDNESS-PQ', 'Exact', 'Bolt',
#                         'FastJL', 'HashJL', 'OSNAP', 'PCA', 'SparsePCA',
#                         'Rademacher', 'RandGauss', 'OrthoGauss')
DEFAULT_PLOT_METHODS = (
    'MADDNESS', 'MADDNESS-PQ', 'Exact', 'ScalarQuantize', 'Bolt',
    # 'MADDNESS', 'Exact', 'ScalarQuantize', 'Bolt',
    # 'FastJL', 'HashJL', 'PCA', 'RandGauss', 'SparsePCA')
    'FastJL', 'HashJL', 'PCA', 'SparsePCA')
    # 'FastJL', 'HashJL', 'PCA', 'SparsePCA')
    # 'MADDNESS', 'MADDNESS-PQ', 'Exact', 'Bolt',
    # 'FastJL', 'HashJL', 'PCA', 'RandGauss', 'SparsePCA')


def lineplot(data, ax, x_metric, y_metric, units=None, scatter=False,
             # plot_methods=None):
             plot_methods=DEFAULT_PLOT_METHODS, first_two_same_marker=True,
             **kwargs):
    estimator = 'mean' if units is None else None
    if plot_methods is not None:
        data = data.loc[data['method'].isin(set(plot_methods))]
        order = plot_methods
    else:
        # order = 'Ours Bolt Exact PQ SVD FD-AMM CD'.split()
        # order = [m for m in order if m in data['Method'].unique()]
        order = list(data['method'].unique())
        # move_methods_to_front = ['Ours', 'OursPQ', 'Brute Force']
        # move_methods_to_front = ['Mithral', 'MithralPQ', 'Brute Force']
        mithral_methods = [method for method in order
                           # if method.lower().startswith('mithral')][::-1]
                           if method.lower().startswith('maddness')][::-1]
        move_methods_to_front = mithral_methods[:]
        # move_methods_to_front.append('Brute Force')
        move_methods_to_front.append('Exact')
        for elem in move_methods_to_front[:]:
            if elem in order:
                order.remove(elem)
            else:
                move_methods_to_front.remove(elem)
        order = move_methods_to_front + sorted(order)

    order = [method for method in order if method in data['method'].unique()]

    # order = plot_methods

    # order = list(data['method'].unique())

    # have to specify markers or seaborn freaks out because it doesn't
    # have enough of them
    # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h',
    #                   'H', 'D', 'd', 'P', 'X')
    # use_markers = ('*', '*', 's') + (
    initial_markers = ('D', 'D', 's') if first_two_same_marker else ('D', 's')
    use_markers = initial_markers + (
        'o', 'v', '^', '<', '>', '8', 'p', 'h', 'd', 'P', 'X', '*', 'D')
    if scatter:
        # sb.violinplot(cut=0, saturation=1, linewidth=.001, scale='width', inner='box',
        # data['Speedup'] *= 1 + (np.random.randn(len(data['Speedup'])) / 100)
        sb.scatterplot(alpha=.25, # seems to suck the least
            data=data, x=x_metric, y=y_metric, hue='method',
            style='method', style_order=order, hue_order=order,
            markers=use_markers, estimator=estimator,
            # units=units, estimator=estimator, markers=use_markers,
            palette=my_colors_list, ax=ax)
        # sb.boxplot(linewidth=.1, width=2, whis=999,
        # sb.stripplot(alpha=.25, orient='v', jitter=False,
        #     data=data, x=x_metric, y=y_metric, hue='method', hue_order=order,
        #     palette=my_colors_list, ax=ax)
        return
    kwargs.setdefault('ci', 'sd')
    sb.lineplot(data=data, x=x_metric, y=y_metric, hue='method',
                # style='method', style_order=order[::-1], hue_order=order[::-1],
                style='method', style_order=order, hue_order=order,
                markers=use_markers, estimator=estimator,
                # units=units, estimator=estimator, markers=use_markers,
                dashes=False, palette=my_colors_list, ax=ax, **kwargs)
    lines = ax.get_lines()
    for i, line in enumerate(lines):
        line.set_zorder(10 - i)


# def cifar_fig(save=False, x_metric='Throughput', y_metric='Accuracy'):
def cifar_fig(save=False, x_metric='Speedup', y_metric='Accuracy'):
    df10 = res.cifar10_amm()
    df100 = res.cifar100_amm()
    sb.set_context('poster')
    # fig, axes = plt.subplots(2, 1, figsize=(11, 13.5), sharex=True)
    # fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)

    # plot_methods = ['Mithral', 'MithralPQ', 'Brute Force', 'Bolt',
    # plot_methods = ['MADDNESS', 'MADDNESS-PQ', 'Exact', 'Bolt',
    #                 'FastJL', 'HashJL', 'OSNAP', 'PCA', 'SparsePCA',
    #                 'Rademacher', 'RandGauss', 'OrthoGauss']
    # # df10 = df10.loc[df10['method'].isin(set(plot_methods))]
    # df100 = df100.loc[df100['method'].isin(set(plot_methods))]

    # df10 = df10.loc[df10['method'] != 'OrthoGauss']
    # df100 = df100.loc[df100['method'] != 'OrthoGauss']

    lineplot(df10, axes[0], x_metric=x_metric, y_metric=y_metric)
    lineplot(df100, axes[1], x_metric=x_metric, y_metric=y_metric)

    # plt.suptitle('Sketch size vs Classification Accuracy')
    xlbl = _xlabel_for_xmetric(x_metric)
    # plt.suptitle('{} vs {}'.format(xlbl, y_metric))
    plt.suptitle('Approximating Softmax Classifiers', family=USE_FONT)
    axes[0].set_title('CIFAR-10', family=USE_FONT)
    for ax in axes:
        ax.set_ylabel(_ylabel_for_xmetric(y_metric), family=USE_FONT)
    axes[0].set_xlabel(None)
    axes[1].set_xlabel(xlbl, family=USE_FONT)
    axes[1].set_title('CIFAR-100', family=USE_FONT)

    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]  # rm 'Method' title
    # axes[0].legend(handles, labels, fontsize='small')
    # axes[1].legend(handles, labels, fontsize='small')
    # plt.figlegend(handles, labels, loc='lower center', ncol=1)
    # plt.figlegend(handles, labels, loc='center right', ncol=1)
    for ax in axes.ravel():
        ax.get_legend().remove()
    if y_metric == 'Accuracy':
        axes[0].set_ylim([.09, .96])
        axes[1].set_ylim([.009, .73])
    elif y_metric == '1 - NMSE':
        axes[0].set_ylim([0, 1.02])
        axes[1].set_ylim([0, 1.02])

    # axes[1].get_legend().remove()
    # axes[1].get_legend().remove()

    plt.figlegend(handles, labels, loc='lower center', ncol=3)

    # if x_metric in ('muls', 'ops', 'nlookups', 'Latency', 'Throughput'):
    axes[0].semilogx()

    for ax in axes:
        if x_metric == 'Speedup':
            ax.set_xlim([.94, ax.get_xlim()[1]])
        elif x_metric == 'NormalizedTime':
            ax.set_xlim([ax.get_xlim()[0], 1.06])

    plt.tight_layout()
    # plt.subplots_adjust(top=.91, bottom=.24)
    plt.subplots_adjust(top=.89, bottom=.32)
    # plt.subplots_adjust(top=.95, bottom=.1)
    save_fig('cifar_{}_{}'.format(x_metric, y_metric))
    # save_fig('cifar_{}_{}_no_maddnesspq'.format(x_metric, y_metric))


def fig1(save=False, x_metric='Speedup', y_metric='Accuracy'):
    df10 = res.cifar10_amm()
    df100 = res.cifar100_amm()
    sb.set_context('poster')
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

    # df10['method'] = df10['method'].str.replace('Mithral', 'HashMul')
    # replace_names_dict = {'Mithral': 'Ours',
    replace_names_dict = {'MADDNESS': 'Ours',
                          # 'SparsePCA': '2nd best (Mairal et al.)',
                          # 'HashJL': '3rd best (Dasgupta et al.)',
                          'SparsePCA': 'Mairal et al.',
                          'HashJL': 'Dasgupta et al.',
                          'Exact': 'Exact Matrix Multiply'
                          }
    # print("--- about to run the rename we care about")
    df10 = res.rename_values_in_col(df10, 'method', replace_names_dict)
    df100 = res.rename_values_in_col(df100, 'method', replace_names_dict)
    # df10['method'] = df10['method'].str.replace(replace_names_dict)
    # df100['method'] = df100['method'].str.replace(replace_names_dict)

    # print('df10 methods: ', df10['method'].unique())
    # import sys; sys.exit()

    # plot_methods = ['Ours', '2nd best', '3rd best', 'Exact Matrix Multiply']
    # plot_methods = ['Ours', 'Mairal et al.', 'Dasgupta et al.', 'Exact Matrix Multiply']
    plot_methods = ['Ours', 'Exact Matrix Multiply', 'Mairal et al.', 'Dasgupta et al.']
    # plot_methods = ['Ours', '3rd best', '2nd best', 'Exact Matrix Multiply']
    # plot_methods = ['Mithral', 'SparsePCA', 'HashJL', 'Brute Force']
    # df10 = df10.loc[df10['method'].isin(set(plot_methods))]
    # df100 = df100.loc[df100['method'].isin(set(plot_methods))]

    # df10 = df10.loc[df10['method'] != 'OrthoGauss']
    # df100 = df100.loc[df100['method'] != 'OrthoGauss']

    lineplot(df10, axes[0], x_metric=x_metric, y_metric=y_metric,
             plot_methods=plot_methods, ci=None, first_two_same_marker=False)
    lineplot(df100, axes[1], x_metric=x_metric, y_metric=y_metric,
             plot_methods=plot_methods, ci=None, first_two_same_marker=False)

    # plt.suptitle('Sketch size vs Classification Accuracy')
    xlbl = _xlabel_for_xmetric(x_metric)
    # plt.suptitle('{} vs {}'.format(xlbl, y_metric))
    plt.suptitle('Approximating Softmax Classifiers', family=USE_FONT)
    axes[0].set_title('CIFAR-10', family=USE_FONT)
    for ax in axes:
        ax.set_ylabel(_ylabel_for_xmetric(y_metric), family=USE_FONT)
    axes[0].set_xlabel(None)
    axes[1].set_xlabel(xlbl, family=USE_FONT)
    axes[1].set_title('CIFAR-100', family=USE_FONT)

    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]  # rm 'Method' title
    # axes[0].legend(handles, labels, fontsize='small')
    # axes[1].legend(handles, labels, fontsize='small')
    # plt.figlegend(handles, labels, loc='lower center', ncol=1)
    # plt.figlegend(handles, labels, loc='center right', ncol=1)
    for ax in axes.ravel():
        ax.get_legend().remove()
    if y_metric == 'Accuracy':
        axes[0].set_ylim([.09, .96])
        axes[1].set_ylim([.009, .73])
    elif y_metric == '1 - NMSE':
        axes[0].set_ylim([0, 1.02])
        axes[1].set_ylim([0, 1.02])

    # axes[1].get_legend().remove()
    # axes[1].get_legend().remove()

    plt.figlegend(handles, labels, loc='lower center', ncol=2)

    # if x_metric in ('muls', 'ops', 'nlookups', 'Latency', 'Throughput'):
    axes[0].semilogx()

    for ax in axes:
        if x_metric == 'Speedup':
            ax.set_xlim([.94, ax.get_xlim()[1]])
        elif x_metric == 'NormalizedTime':
            ax.set_xlim([ax.get_xlim()[0], 1.06])

    plt.tight_layout()
    plt.subplots_adjust(top=.89, bottom=.23)
    save_fig('fig1')


def caltech_fig(x_metric='Speedup', y_metric='1 - NMSE'):
    # df = res.caltech_amm()
    # df = res.caltech_amm()
    df0 = res.caltech_amm(filt='sobel')
    df1 = res.caltech_amm(filt='dog5x5')
    # print("df cols: ", df.columns)

    sb.set_context('poster')
    # fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    # axes = [ax]
    # is_mithral = df['method'].str.startswith('Mithral')
    # is_exact = df['method'] == 'Brute Force'
    # others_to_keep = df['method'].isin(['Brute Force', 'PCA', 'SparsePCA'])
    # others_to_keep = df['method'].isin(['PCA', 'SparsePCA'])
    # df = df.loc[is_mithral | others_to_keep]  # others suck too hard

    # df = df.loc[~(df['method'].isin(['Mithral, L = 2', 'Mithral, L = 4']))]
    # df['method'].loc[df['method'] == 'Mithral, L = ∞'] = 'Mithral'

    # print("df0 uniq methods: ", df0['method'].unique())
    # print("df1 uniq methods: ", df1['method'].unique())
    # import sys; sys.exit()

    # keep_methods = ['Mithral', 'MithralPQ', 'SparsePCA', 'PCA', 'OSNAP']
    # keep_methods = ['Mithral', 'MithralPQ', 'SparsePCA', 'PCA', 'HashJL', 'OSNAP', 'FastJL']
    # keep_methods = ['Mithral', 'MithralPQ', 'SparsePCA', 'PCA']
    # keep_methods = ['MADDNESS', 'MADDNESS-PQ', 'SparsePCA', 'PCA']

    # even scalar quantize is slower than custom exact matmul; note that
    # in the 5x5 plot, it's occluded by maddness (near perfect mse, but
    # slightly to the left of 1x speedup)
    # keep_methods = ['MADDNESS', 'MADDNESS-PQ', 'ScalarQuantize', 'SparsePCA']
    keep_methods = ['MADDNESS', 'MADDNESS-PQ', 'SparsePCA']
    df0 = df0.loc[df0['method'].isin(keep_methods)]
    df1 = df1.loc[df1['method'].isin(keep_methods)]

    # print("df0 kept methods: ", df0['method'].unique())
    # print("df1 kept methods: ", df1['method'].unique())
    # print("df1 scalar quantize numbers: ", df1.loc[df1['method'] == 'ScalarQuantize'])
    # import sys; sys.exit()

    # print("df1:\n", df1.loc[(df1['method'] == 'MithralPQ') & df1['task_id'].str.contains('509')])
    # import sys; sys.exit()

    # lineplot(df, ax, x_metric=x_metric, y_metric=y_metric, units=None)
    lineplot(df0, axes[0], x_metric=x_metric, y_metric=y_metric,
             plot_methods=keep_methods)
    lineplot(df1, axes[1], x_metric=x_metric, y_metric=y_metric,
             plot_methods=keep_methods)

    handles, labels = axes[-1].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]  # rm 'Method' title
    # plt.figlegend(handles, labels, loc='lower center', ncol=2)
    # plt.figlegend(handles, labels, loc='lower center', ncol=4)
    plt.figlegend(handles, labels, loc='lower center', ncol=len(keep_methods))

    # plt.suptitle('Approximating an Image Filter')
    for ax in axes:
        ax.set_xlabel(_xlabel_for_xmetric(x_metric), fontsize=20)
        ax.set_ylabel(y_metric)
        ax.get_legend().remove()
        ax.set_ylim([-.01, 1.01])
        ax.plot([1, 1], ax.get_ylim(), 'k--')
    # for ax in axes[:-1]:
    #     # remove x labels except for bottom axis
    #     plt.setp(ax.get_xticklabels(), visible=False)
    #     ax.get_xaxis().set_visible(False)
    axes[0].set_title('Approximating a Sobel Filter', y=1.02, fontsize=28)
    axes[1].set_title('Approximating a Gaussian Filter', y=1.02, fontsize=28)

    # plt.subplots_adjust(top=.91, bottom=.37)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=.26, hspace=.72)  # with ncol=2
    plt.subplots_adjust(bottom=.22, hspace=.7)  # with ncol=2
    # plt.subplots_adjust(top=.95, bottom=.1)
    save_fig('caltech_{}_{}'.format(x_metric, '1 - NMSE'))
    # save_fig('caltech_sobel_{}_{}'.format(x_metric, '1 - NMSE'))
    # save_fig('caltech_dog_{}_{}'.format(x_metric, '1 - NMSE'))


# def ucr_fig(x_metric='Speedup', y_metric='Accuracy'):
# def ucr_fig(x_metric='Speedup', y_metric='Change in Accuracy'):
def ucr_fig(x_metric='Speedup', y_metric='Relative Accuracy'):
    # df = res.ucr_amm()
    # df = res.ucr_amm(k=64)
    # df = res.ucr_amm(k=128)
    # df = res.ucr_amm(k=256)
    df0 = res.ucr_amm(k=64)
    df1 = res.ucr_amm(k=128)
    df2 = res.ucr_amm(k=256)
    sb.set_context('poster')
    # fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    # axes = [ax]

    # df = df.loc[df['task_id'].str.lower().str.contains('starlight')]
    # df = df.loc[df['method'] == 'Mithral']
    # # df = df.loc[df['method'] == 'MithralPQ']
    # # df = df.loc[df['ncodebooks'] == 4]
    # df = df['Accuracy acc_orig acc_orig_1nn ncodebooks method task_id'.split() + ['Relative Accuracy']]
    # df.reset_index(inplace=True, drop=True)
    # print(df)
    # import sys; sys.exit()

    # df['Change in Accuracy'] = df['Accuracy'] - df['acc-1nn-raw']

    # print("uniq N, D, M: ")
    # print(df['N'].unique())
    # print(df['D'].unique())
    # print(df['M'].unique())
    # df_brute = df.loc[df['method'] == 'Brute Force']
    # print("uniq times from brute force: ", df_brute['time'].unique())
    # print("df Brute:\n", df_brute['N D M method normalized_mse Accuracy time'.split()])
    # import sys; sys.exit()

    # df['acc']

    # # TODO put in results cleaning after debug
    # if 'Accuracy' in df.columns:
    #     # df['Relative Accuracy'] = df['Accuracy'] / (df['acc_orig'] + 1e-20)
    #     # # note that relative accuracy can actually be higher if errors
    #     # # happen to compensate for incorrect classification sometimes
    #     # print("max relative acc: ", df['Relative Accuracy'].values.max())
    #     # # assert df['Relative Accuracy'].values.max() <= 1.000001

    #     # acc_orig field is supposed to capture this, but I messed it up for
    #     # 1nn so this will also work
    #     tid2acc = {}
    #     exactdf = df.loc[df['method'] == 'Brute Force']
    #     for tid in df['task_id'].unique():
    #         subdf = exactdf.loc[exactdf['task_id'] == tid]
    #         if subdf.shape[0] != 1:
    #             print(f"tid = {tid} gives bad subdf:\n", subdf)
    #         tid2acc[tid] = subdf['Accuracy'].values[0]
    #     df['BaseAccuracy'] = [tid2acc[tid] for tid in df['task_id']]
    #     df['Relative Accuracy'] = df['Accuracy'] / df['BaseAccuracy']

    # df = df.loc[~(df['method'].isin(['Mithral, L = 2', 'Mithral, L = 4']))]
    # # df['method'].loc[df['method'] == 'Mithral, L = ∞'] = 'Mithral'
    # df0 = df0.loc[df0['method'] != 'Brute Force']
    # df1 = df1.loc[df1['method'] != 'Brute Force']
    # df2 = df2.loc[df2['method'] != 'Brute Force']

    # print(df.columns)
    # import sys; sys.exit()

    def clean_df(df):
        df['Change in Accuracy'] = df['Accuracy'] - df['acc-1nn-raw']
        # is_mithral = df['method'].str.startswith('Mithral')
        # is_mithral = df['method'] == 'Mithral'
        is_mithral = df['method'] == 'MADDNESS'
        # # is_exact = df['method'] == 'Brute Force'
        others_to_keep = df['method'].isin([
            'PCA', 'SparsePCA', 'Bolt', 'HashJL', 'OSNAP'])
        # others_to_keep = df['method'].isin(['PCA', 'SparsePCA'])
        return df.loc[is_mithral | others_to_keep]

    df0 = clean_df(df0)
    df1 = clean_df(df1)
    df2 = clean_df(df2)

    # df = df.loc[df['method'] == 'Brute Force']

    # df['not_mse'] = 1. - df['normalized_mse']
    # df = df.loc[df['not_mse'] < 2]
    lineplot(df0, axes[0], x_metric=x_metric, y_metric=y_metric, scatter=True)
    lineplot(df1, axes[1], x_metric=x_metric, y_metric=y_metric, scatter=True)
    lineplot(df2, axes[2], x_metric=x_metric, y_metric=y_metric, scatter=True)

    plt.suptitle('Approximating an RBF Kernel Classifier')
    axes[-1].set_xlabel(_xlabel_for_xmetric(x_metric))
    # ax.set_ylabel('1. - NMSE')

    handles, labels = axes[-1].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]  # rm 'Method' title
    plt.figlegend(handles, labels, loc='lower center', ncol=3)

    for ax in axes:
        ax.set_ylabel(_ylabel_for_xmetric(y_metric))
        ax.get_legend().remove()
        ax.semilogx()
        ax.set_xlim([.9, ax.get_xlim()[1]])

    # ax.set_ylim([.2, 1.1])
    # plt.plot([1, 1], ax.get_ylim(), 'k--')

    plt.tight_layout()
    plt.subplots_adjust(top=.94, bottom=.25)
    # plt.subplots_adjust(top=.95, bottom=.1)
    save_fig('ucr_{}_{}'.format(x_metric, y_metric))


def ucr_fig2(x_metric='Speedup', y_metric='Relative Accuracy',
             # problem='softmax'):
             problem='rbf'):
    # df0 = res.ucr_amm(k=64)
    # df1 = res.ucr_amm(k=128)
    # df2 = res.ucr_amm(k=256)
    df = res.ucr_amm(k=128, problem=problem)
    sb.set_context('poster')
    # fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # df = res.ucr_amm(k=128, problem='rbf')
    # df_bolt = df.loc[df['method'] == 'Bolt']
    # print("number of uniq bolt speedups:")
    # print(df_bolt['Speedup'].unique().size)
    # import sys; sys.exit()

    def clean_df(df):
        df['Change in Accuracy'] = df['Accuracy'] - df['acc-1nn-raw']
        return df

        # # is_mithral = df['method'].str.startswith('Mithral')
        # is_mithral = df['method'] == 'Mithral'
        # # # is_exact = df['method'] == 'Brute Force'
        # others_to_keep = df['method'].isin([
        #     'PCA', 'SparsePCA', 'Bolt', 'HashJL', 'OSNAP'])
        # # others_to_keep = df['method'].isin(['PCA', 'SparsePCA'])
        # return df.loc[is_mithral | others_to_keep]

    def frac_above_thresh(df, thresh):
        return res.frac_above_thresh(
            df, x_metric, y_metric, 'method', 'task_id', thresh)

    df = clean_df(df)
    # df0['frac_above_thresh'] = frac_above_thresh(df, .5)

    # df_bolt = df.loc[df['method'] == 'Bolt']
    # print("number of uniq bolt speedups:")
    # print(df_bolt['Speedup'].unique().size)
    # import sys; sys.exit()

    # df = df.loc[df['method'] == 'SparsePCA']
    # print(df.groupby('task_id')['Speedup'].count())
    # import sys; sys.exit()

    y_frac_thresholds = [.5, .75, .95]
    df0 = frac_above_thresh(df, y_frac_thresholds[0])
    df1 = frac_above_thresh(df, y_frac_thresholds[1])
    df2 = frac_above_thresh(df, y_frac_thresholds[2])

    # # print(df0['frac_above_thresh'])
    # print(df0)
    # # for row in df0.iterrows():
    # #     print(row)
    # # print(df0.unstack(0))
    # print("df cols: ", df.columns)
    # print("df0 cols: ", df0.columns)
    # print("uniq methods: ", df['method'].unique())

    # df = df.loc[df['method'] == 'Brute Force']

    # df['not_mse'] = 1. - df['normalized_mse']
    # df = df.loc[df['not_mse'] < 2]
    ycol = 'frac_above_thresh'
    lineplot(df0, axes[0], x_metric=x_metric, y_metric=ycol, scatter=False)
    lineplot(df1, axes[1], x_metric=x_metric, y_metric=ycol, scatter=False)
    lineplot(df2, axes[2], x_metric=x_metric, y_metric=ycol, scatter=False)

    kind = 'a Softmax' if problem == 'softmax' else 'an RBF Kernel'
    plt.suptitle(f'Approximating {kind} Classifier')
    axes[-1].set_xlabel(_xlabel_for_xmetric(x_metric))
    # ax.set_ylabel('1. - NMSE')

    handles, labels = axes[-1].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]  # rm 'Method' title
    plt.figlegend(handles, labels, loc='lower center', ncol=3)

    for i, ax in enumerate(axes):
        # ax.set_ylabel(_ylabel_for_xmetric(y_metric))
        # ax.set_ylabel("Fraction of Datasets\nWith Relative Acc > "
        #               f"{y_frac_thresholds[i]}")
        # ax.set_ylabel(f"Fraction with Relative\nAccuracy> {y_frac_thresholds[i]}")
        ax.set_ylabel(f"Fraction > {y_frac_thresholds[i]}")
        ax.get_legend().remove()
        ax.semilogx()
        ax.set_xlim([.9, ax.get_xlim()[1]])
        ax.set_ylim([0, 1.03])

    # ax.set_ylim([.2, 1.1])
    # plt.plot([1, 1], ax.get_ylim(), 'k--')

    plt.tight_layout()
    # plt.subplots_adjust(top=.94, bottom=.25)
    plt.subplots_adjust(top=.94, bottom=.22)
    # plt.subplots_adjust(top=.95, bottom=.1)
    save_fig('ucr2_{}_{}_{}'.format(x_metric, y_metric, problem))


def main():
    scan_speed_fig()
    encode_speed_fig()
    lut_speed_fig()
    fig1()
    ucr_fig2()
    caltech_fig()
    # caltech_fig(y_metric='1 - NMSE')
    # caltech_fig(x_metric='ops', y_metric='1 - NMSE')
    cifar_fig()
    # cifar_fig(y_metric='1 - NMSE')
    # cifar_fig(x_metric='ops')
    # cifar_fig(x_metric='ops', y_metric='1 - NMSE')
    # ucr_fig2(x_metric='ops', y_metric='1 - NMSE')
    # ucr_fig2(x_metric='ops')
    # cifar_fig(y_metric='1 - NMSE')
    # ucr_fig2()
    # ucr_fig2(y_metric='1 - NMSE')


if __name__ == '__main__':
    main()
