
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

import results


SAVE_DIR = os.path.expanduser('~/Desktop/bolt/figs/')


def save_fig(name):
    plt.savefig(os.path.join(SAVE_DIR, name + '.pdf'), bbox_inches='tight')


def set_palette(ncolors=8):  # use this to change color palette in all plots
    pal = sb.color_palette("Set1", n_colors=8)
    sb.set_palette(pal)

    return pal


def popcount_fig(fake_data=False):

    # sb.set_context("poster", rc={"figure.figsize": (8, 4)})
    sb.set_context("talk")
    set_palette(ncolors=2)
    _, ax = plt.subplots(1, figsize=(6, 4))

    # fake_data = data is None
    if fake_data:  # for prototyping / debugging this func
        bolt_128d_8b = np.random.randn(10) + 12
        bolt_128d_16b = np.random.randn(10) + 6
        bolt_128d_32b = np.random.randn(10) + 3
        popcnt_128d_8b = np.random.randn(10) + 10
        popcnt_128d_16b = np.random.randn(10) + 5
        popcnt_128d_32b = np.random.randn(10) + 2

        dicts = []
        dicts += [{'algo': 'Bolt', 'nbytes': '8B',  't': t} for t in bolt_128d_8b]
        dicts += [{'algo': 'Bolt', 'nbytes': '16B',  't': t} for t in bolt_128d_16b]
        dicts += [{'algo': 'Bolt', 'nbytes': '32B',  't': t} for t in bolt_128d_32b]
        dicts += [{'algo': 'Binary Embedding', 'nbytes': '8B',  't': t} for t in popcnt_128d_8b]
        dicts += [{'algo': 'Binary Embedding', 'nbytes': '16B',  't': t} for t in popcnt_128d_16b]
        dicts += [{'algo': 'Binary Embedding', 'nbytes': '32B',  't': t} for t in popcnt_128d_32b]

        df = pd.DataFrame.from_records(dicts)

    else:
        df = results.popcount_results()

    print "df cols: ", df.columns
    # df.rename(columns={'algo': 'Algorithm'}, inplace=True)
    df.rename(columns={'algo': ' '}, inplace=True)  # hide from legend

    sb.barplot(x='nbytes', y='t', hue=' ', ci=95, data=df, ax=ax)
    ax.set_title('Distance Computations Per Second')
    ax.set_xlabel('Encoding Length (Bytes)')
    ax.set_ylabel('Millions of distances / sec')

    plt.tight_layout()
    save_fig('popcount_speed')
    # plt.show()


def encoding_fig(fake_data=False):
    # sb.set_context("talk", rc={"figure.figsize": (6, 6)})
    sb.set_context("talk", rc={"figure.figsize": (7, 7)})
    # sb.set_context("talk", rc={"figure.figsize": (8, 8)})
    # sb.set_context("talk", rc={"figure.figsize": (9, 9)})

    # fig, axes = plt.subplots(3, 1)
    fig, axes = plt.subplots(3, 2)

    # ALGOS = ['Bolt', 'PQ', 'OPQ', 'PairQ']
    ALGOS = ['Bolt', 'PQ', 'OPQ']
    algo2offset = {'Bolt': 100, 'PQ': 50, 'OPQ': 30, 'PairQ': 25}
    lengths = [64, 128, 256, 512, 1024]
    # results_for_algos_lengths =

    # sb.set_palette("Set1", n_colors=len(ALGOS))
    set_palette(ncolors=len(ALGOS))

    if fake_data:
        data = np.random.randn(1, len(lengths), len(algo2offset))
        for i, algo in enumerate(ALGOS):
            data[:, :, i] += algo2offset[algo]
        data /= np.arange(len(lengths)).reshape((1, -1, 1))

        # ------------------------ data encoding

        # 8B encodings
        ax = axes[0, 0]
        # sb.tsplot(data=data, condition=condition, time=lengths, ax=ax)
        sb.tsplot(data=data, condition=None, time=lengths, ax=ax)
        # ax.set_title(prefix + ' Encoding Speed, 8B codes')
        ax.set_title('Data Encoding Speed', y=1.02)

        # 16B encodings
        data /= 2
        ax = axes[1, 0]
        sb.tsplot(data=data, condition=None, time=lengths, ax=ax)

        # 32B encodings
        data /= 2
        ax = axes[2, 0]
        sb.tsplot(data=data, condition=None, time=lengths, ax=ax)

        # ------------------------ query encoding
        data *= 8
        data += np.random.randn(*data.shape) * 5

        # 8B encodings
        ax = axes[0, 1]
        sb.tsplot(data=data, condition=None, time=lengths, ax=ax)
        # ax.set_title(prefix + ' Encoding Speed')
        ax.set_title('Query Encoding Speed', y=1.03, fontsize=16)

        # 16B encodings
        data /= 2
        ax = axes[1, 1]
        sb.tsplot(data=data, condition=None, time=lengths, ax=ax)

        # 32B encodings
        data /= 2
        ax = axes[2, 1]
        sb.tsplot(data=data, condition=ALGOS, time=lengths, ax=ax)

    else:  # real data
        NBYTES_LIST = [8, 16, 32]

        df = results.encode_results()
        df_x = df[df['task'] == 'encode_x']
        df_q = df[df['task'] == 'encode_q']
        dfs = [df_x, df_q]

        # print df_x
        # return

        # dfs = [results.encode_data_results(), results.encode_lut_results()]
        ax_cols = [axes[:, 0], axes[:, 1]]

        for df, ax_col in zip(dfs, ax_cols):  # for each col in subplots
            for b, nbytes in enumerate(NBYTES_LIST):  # for each row in subplots
                ax = ax_col[b]
                plot_df = df.loc[df['nbytes'] == nbytes]
                plot_df = plot_df.loc[plot_df['algo'].isin(ALGOS)]
                sb.tsplot(value='y', condition='algo', unit='trial', time='D',
                          data=plot_df, ax=ax, ci=95, n_boot=500)
                    # data=plot_df, ax=ax, legend=False, ci=95, n_boot=500)

    # ------------------------ legend

    ax = axes.ravel()[-1]
    leg_lines, leg_labels = ax.get_legend_handles_labels()
    # ax.legend_.remove()
    # leg_lines, leg_labels = leg_lines[:len(ALGOS)], leg_labels[:len(ALGOS)]

    plt.figlegend(leg_lines, leg_labels, loc='lower center',
                  ncol=len(ALGOS), labelspacing=0)

    # ------------------------ postproc + save plot

    for ax in axes.ravel():
        ax.set_yscale("log")
        ax.legend_.remove()
        ax.set_ylim(5e3, 2e7)

    axes[0, 0].set_title('Data Encoding Speed', y=1.03, fontsize=16)
    axes[0, 1].set_title('Query Encoding Speed', y=1.03, fontsize=16)
    # for ax in axes[0, :].ravel():
        # ax.set_title('Vector Length')

    for ax in axes[:-1, :].ravel():
        # ax.xaxis.set_visible(False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel('', labelpad=-10)
    for ax in axes[-1, :].ravel():
        ax.set_xlabel('Vector Length')
    for ax in axes[:, 0]:
        ax.set_ylabel('Vectors Encoded / s')

    # only bottom row gets xlabels
    for ax in axes[:-1, :].ravel():
        # plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel('', labelpad=-10)

    # show byte counts on the right
    fmt_str = "{}B Encodings"
    for i, ax in enumerate(axes[:, 1].ravel()):
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(fmt_str.format((2 ** i) * 8), labelpad=10, fontsize=15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=.15)
    save_fig('encoding_speed')
    # plt.show()
    # if data_enc:
    #     save_fig('encoding_speed_data')
    # else:
    #     save_fig('encoding_speed_query')
    # plt.show()


def query_speed_fig(fake_data=False, fname='query_speed', with_matmuls=True):
    # experiment params: fixed N = 100k, D = 256, Q = 1024;
    # layout: rows = 8B, 16B, 32B; bar graph in each row
    #   alternative: plot in each row vs batch size
    # algos: Bolt; PQ; OPQ; PairQ; Matmul, batch={1, 16, 64, 256}

    sb.set_context("talk")
    set_palette(ncolors=8)
    fig, axes = plt.subplots(3, 1, figsize=(6, 8))

    if fake_data:  # for debugging
        ALGOS = ['Bolt', 'PQ', 'OPQ', 'PairQ',
                 # 'Matmul Batch 1', 'Matmul Batch 16', 'Matmul Batch 64', 'Matmul Batch 256']
                 # 'Matmul Batch1', 'Matmul Batch16', 'Matmul Batch64', 'Matmul Batch256']
                 'Matmul 1', 'Matmul 16', 'Matmul 64', 'Matmul 256']
        algo2offset = {'Bolt': 100, 'PQ': 50, 'OPQ': 30, 'PairQ': 25,
                       # 'Matmul Batch 1': 1, 'Matmul Batch 16': 16,
                       # 'Matmul Batch 64': 64, 'Matmul Batch 256': 256}
                       # 'Matmul Batch1': 1, 'Matmul Batch16': 16,
                       # 'Matmul Batch64': 64, 'Matmul Batch256': 256}
                       'Matmul 1': 1, 'Matmul 16': 16, 'Matmul 64': 64,
                       'Matmul 256': 256}

        for i, nbytes in enumerate([8, 16, 32]):
            bytes_str = '{}B'.format(nbytes)
            dicts = []
            for algo in ALGOS:
                dps = np.random.randn(10) + 256 / nbytes
                dps += algo2offset[algo] / nbytes
                dicts += [{'algo': algo, 'nbytes': bytes_str, 'y': y} for y in dps]

            df = pd.DataFrame.from_records(dicts)
    else:
        # ALGOS = ['Bolt', 'PQ', 'OPQ', 'PairQ', 'Matmul 1', # 'Matmul 16',
        #          'Matmul 64', 'Matmul 256', 'Matmul 1024']

        if with_matmuls:
            ALGOS = ['Bolt', 'Binary Embedding', 'PQ', 'OPQ',
                    'Matmul 1', 'Matmul 256', 'Matmul 1024']
        else:
            ALGOS = ['Bolt', 'Binary Embedding', 'PQ', 'OPQ']
        df = results.query_speed_results()
        df['y'] = df['y'] / 1e9  # convert to billions

    print "df cols: ", df.columns
    df.rename(columns={'algo': ' '}, inplace=True)  # hide from legend

    # ax = sb.barplot(x='x', y='y', hue=' ', ci=95, data=df, ax=axes[i])
    for i, nbytes in enumerate([8, 16, 32]):
        bytes_str = '{}B'.format(nbytes)
        data = df[df['nbytes'] == nbytes]
        ax = sb.barplot(x='nbytes', y='y', hue=' ', hue_order=ALGOS, ci=95, data=data, ax=axes[i])

    # ------------------------ clean up / format axes

    for ax in axes[:-1]:
        # remove x labels except for bottom axis
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.get_xaxis().set_visible(False)

    end = .5 * (len(ALGOS) / float((len(ALGOS) + 2)))
    start = -end
    tick_positions = np.linspace(start + .02, end - .05, len(ALGOS))

    for ax in axes:
        ax.set_xlim([start - .02, end + .02])
        ax.set_ylabel('Billions of Distances/s')
        ax.legend_.remove()
        if not fake_data:
            ax.set_ylim(0, 2.5)

    # add byte counts on the right
    sb.set_style("white")  # adds border (spines) we have to remove
    fmt_str = "{}B Encodings"
    for i, ax in enumerate(axes):
        ax2 = ax.twinx()
        sb.despine(ax=ax2, top=True, left=True, bottom=True, right=True)
        ax2.get_xaxis().set_visible(False)
        # ax2.get_yaxis().set_visible(False)  # nope, removes ylabel
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel(fmt_str.format((2 ** i) * 8), labelpad=10, fontsize=15)

    # ------------------------ have bottom / top axes print title, x info

    axes[0].set_title('Distance Computations per Second', y=1.02)

    # axes[-1].set_xticks(tick_positions)
    for ax in axes:
        axes[-1].set_xticks(tick_positions)
        ax.set_xlim(-.4, .4)  # no idea why this makes the bars fit right...
    xlabels = ["\n".join(name.split(' ')) for name in ALGOS]
    for i, lbl in enumerate(xlabels):
        if '\n' in lbl:
            # shift label up by adding another line
            xlabels[i] = xlabels[i] + '\n'
    # xlabels = ["\nBatch".join(name.split(' Batch')) for name in ALGOS]
    # xlabels = ALGOS
    axes[-1].set_xticklabels(xlabels, rotation=70)
    axes[-1].set_xlabel("", labelpad=-20)
    # plt.setp(axes[-1].get_xlabel(), visible=False)  # doesn't work

    # ------------------------ show / save plot

    plt.tight_layout()
    save_fig(fname)
    # plt.show()


def matmul_fig(fake_data=False, fname='matmul'):
    # two line graphs
    # lines in both top and bottom = bolt {8,16,32}B, matmul
    # just use square mats of power-of-two lengths cuz best case for matmuls
    # in top one, one mat already encoded and Bolt just has to do queries
    # in bottom one, Bolt has encode one of the mats as data before queries


    sb.set_context("talk")
    # sb.set_palette("Set1", n_colors=len(ALGOS))
    pal = set_palette(ncolors=8)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    # axes = axes.reshape((2, 1))

    if fake_data:  # for debuging / prototyping fig
        SIZES = np.array([64, 128, 256, 512, 1024, 2048, 4096], dtype=np.float32)
        matmul_times = (SIZES ** 2.5).reshape((-1, 1))  # strassen-ish scaling
        bolt_times = ((SIZES ** 3) / 100 + 400).reshape((-1, 1))

        # pretend we had 5 trials; each trial gets a col, so rows are lengths
        matmul_times = np.tile(matmul_times, (1, 5))
        bolt_times = np.tile(bolt_times, (1, 5))
        matmul_times += np.random.randn(*matmul_times.shape) * SIZES.T.reshape((-1, 1)) / 10.
        bolt_times += np.random.randn(*bolt_times.shape) * SIZES.T.reshape((-1, 1)) / 10.

        matmul_times /= 1e9
        bolt8_times = bolt_times / 2e9
        bolt16_times = bolt_times / 1e9
        bolt32_times = bolt_times / .5e9

        dicts = []
        ALGOS = ['Bolt 8B', 'Bolt 16B', 'Bolt 32B', 'Floats']
        algo_times = [bolt8_times, bolt16_times, bolt32_times, matmul_times]
        for all_times, algo in zip(algo_times, ALGOS):
            for sz, times_for_sz in zip(SIZES, all_times):
                dicts += [{'algo': algo, 'trial': i, 'size': sz, 'y': t}
                          for i, t in enumerate(times_for_sz)]

        df = pd.DataFrame.from_records(dicts)
        df_enc = df
        df_no_enc = df

        sb.tsplot(time='size', value='y', condition='algo', unit='trial',
                  data=df_no_enc, ax=axes[0], n_boot=100)
        sb.tsplot(time='size', value='y', condition='algo', unit='trial',
                  data=df_enc, ax=axes[1], n_boot=100)

    else:
        # ALGOS = ['Bolt 8B', 'Bolt 16B', 'Bolt 32B', 'Floats']
        # ALGOS = ['Bolt 32B', 'Bolt 32B + Encode', 'Floats']
        # ALGOS = ['Bolt 8B', 'Bolt 32B', 'Bolt 32B + Encode', 'Floats']
        ALGOS = ['Bolt 8B', 'Bolt 8B + Encode', 'Bolt 32B', 'Bolt 32B + Encode', 'Floats']
        # df = results.matmul_results_square()

        def clean_df(df):
            df = df.loc[df['algo'].isin(ALGOS)]
            non_encode_algos = ['Bolt 8B', 'Bolt 16B', 'Bolt 32B']
            # rm_idxs = (df['algo'] == 'Bolt 32B') * (df['enc'] == 1)
            rm_idxs = (df['algo'].isin(non_encode_algos)) * (df['enc'] == 1)
            return df.loc[~rm_idxs]

        df = results.matmul_results(which='square')
        df = clean_df(df)

        colors = {
            'Bolt 8B': pal[0], 'Bolt 8B + Encode': pal[0],
            # 'Bolt 16B': pal[2], 'Bolt 16B + Encode': pal[2],
            'Bolt 32B': pal[1], 'Bolt 32B + Encode': pal[1],
            'Floats': 'k'
        }
        df_no_enc = df.loc[df['enc'] != 1]
        sb.tsplot(time='size', value='y', condition='algo', unit='trial',
                  data=df_no_enc, ax=axes[0], n_boot=100, color=colors, linestyle='solid')
        df_enc = df.loc[df['enc'] == 1]
        sb.tsplot(time='size', value='y', condition='algo', unit='trial',
                  data=df_enc, ax=axes[0], n_boot=100, color=colors, linestyle='dotted', lw=4)

        df = results.matmul_results(which='tall')
        df = clean_df(df)

        # print df
        # return

        # sb.tsplot(time='size', value='y', condition='algo', unit='trial',
        #           data=df, ax=axes[1], n_boot=100, color=colors)
        df_no_enc = df.loc[df['enc'] != 1]
        sb.tsplot(time='size', value='y', condition='algo', unit='trial',
                  data=df_no_enc, ax=axes[1], n_boot=100, color=colors, linestyle='solid')
        df_enc = df.loc[df['enc'] == 1]
        sb.tsplot(time='size', value='y', condition='algo', unit='trial',
                  data=df_enc, ax=axes[1], n_boot=100, color=colors, linestyle='dotted', lw=4)


        # axes[1].set_ylim(1, 1e3)

    # without encoding at the top; with encoding on the bottom
    # sb.tsplot(time='size', value='y', condition='algo', unit='trial',
    # sb.tsplot(time='size', value='y', condition='algo', unit='trial',
    #     data=df_no_enc, ax=axes[0], n_boot=100)
    # sb.tsplot(time='size', value='y', condition='algo', unit='trial',
    #     data=df_enc, ax=axes[1], n_boot=100)

    # ------------------------ legend

    ax = axes.ravel()[-1]
    leg_lines, leg_labels = ax.get_legend_handles_labels()
    # for some reason, each algo appears 3x, so just take first
    leg_lines, leg_labels = leg_lines[:len(ALGOS)], leg_labels[:len(ALGOS)]

    plt.figlegend(leg_lines, leg_labels, loc='lower center',
                  ncol=len(ALGOS)/2, labelspacing=0)

    # ------------------------ axis cleanup / formatting

    # axes[0].set_title('Matrix Multiply Time, One Matrix Encoded', y=1.03, fontsize=16)
    # axes[1].set_title('Matrix Multiply Time, Neither Matrix Encoded', y=1.03, fontsize=16)
    axes[0].set_title('Square Matrix Multiply Time', y=1.03, fontsize=16)
    axes[1].set_title('Tall Matrix Multiply Time', y=1.03, fontsize=16)

    for ax in axes.ravel():
        ax.legend_.remove()
        ax.set_xscale('log', basex=2)
        ax.set_yscale('log', basey=10)
        ax.set_ylabel('Wall Time (s)')
    # for ax in axes[:-1].ravel():
    # #     plt.setp(ax.get_xticklabels(), visible=False)
    #     ax.set_xlabel('', labelpad=-10)

    # axes[0].set_xlabel('Matrix Side Length, L', labelpad=-1)
    axes[0].set_xlabel('Matrix Side Length, L')
    axes[1].set_xlabel('Matrix Side Length, L')

    # ------------------------ show / save plot

    # plt.tight_layout(h_pad=1.4)
    plt.tight_layout(h_pad=1.2)
    plt.subplots_adjust(bottom=.23)
    save_fig('matmul_speed')
    # plt.show()


def recall_r_fig(fake_data=False, suptitle=None, l2=True, fname='l2_recall'):
    # experiment params:
    #   datasets = Sift1M, Convnet1M, LabelMe22k, MNIST
    #   bytes = [8, 16, 32]
    #   R = 1, 10, 100, 1000

    DATASETS = ['Sift1M', 'Convnet', 'LabelMe', 'MNIST']
    ALGOS = ['Bolt', 'PQ', 'OPQ', 'PairQ']
    NBYTES_LIST = [8, 16, 32]
    # Rs = [1, 10, 100, 1000]
    Rs = [1, 5, 10, 50, 100, 500, 1000]

    sb.set_context("talk")
    set_palette(ncolors=len(ALGOS))
    fig, axes = plt.subplots(4, 3, figsize=(6, 9))

    if suptitle is None:
        suptitle = 'Nearest Neighbor Recall'

    if fake_data:
        algo2offset = {'Bolt': -.1, 'PQ': -.2, 'OPQ': 0, 'PairQ': .1}
        data = np.random.rand(1, len(Rs), len(algo2offset))
        data = np.sort(data, axis=1)  # ensure fake recalls are monotonic
        for i, algo in enumerate(ALGOS):
            recall = data[:, :, i] + algo2offset[algo]
            data[:, :, i] = np.clip(recall, 0., 1.)

        line_styles_for_nbytes = {8: '-', 16: '-', 32: '-'}

        # plot the data
        for d, dataset in enumerate(DATASETS):
            axes_row = axes[d]
            for b, nbytes in enumerate(NBYTES_LIST):
                ax = axes_row[b]
                if fake_data:  # TODO handle real data
                    data_tmp = data * (.5 + nbytes / 64.)  # slightly less
                assert np.max(data_tmp) <= 1.
                for algo in ALGOS:
                    x = Rs
                    sb.tsplot(data=data_tmp, condition=ALGOS, time=x, ax=ax, n_boot=100,
                              ls=line_styles_for_nbytes[nbytes])

    else:  # real data
        DATASETS = ['Sift1M', 'Convnet1M', 'LabelMe', 'MNIST']
        # ALGOS = ['PQ', 'OPQ']
        ALGOS = ['Bolt', 'Bolt No Quantize', 'PQ', 'OPQ']
        for d, dset in enumerate(DATASETS):
            if l2:
                path = os.path.join('../results/recall_at_r/', dset, 'summary.csv')
            else:
                path = os.path.join('../results/recall_at_r_mips/', dset, 'summary.csv')

            df = pd.read_csv(path)
            pq4 = (df['_algo'] == 'PQ') & (df['_code_bits'] == 4)
            df.loc[pq4, '_algo'] = 'Bolt No Quantize'

            # rm results with bolt rotations
            bolt_rot = (df['_algo'] == 'Bolt') & (df['opq_iters'] > 0)
            df = df.loc[~bolt_rot]

            df.rename(columns={'_algo': 'algo'}, inplace=True)
            all_nbytes = (df['_code_bits'] * df['_ncodebooks'] / 8).values
            df['nbytes'] = all_nbytes.astype(np.int)

            for b, nbytes in enumerate(NBYTES_LIST):
                ax = axes[d, b]
                data = df.loc[df['nbytes'] == nbytes]
                for algo in ALGOS:
                    df_row = data.loc[data['algo'] == algo]  # should be 1 row
                    if len(df_row) != 1:
                        print df_row
                        print "dset = ", dset
                        print "algo = ", algo
                        assert len(df_row) == 1
                    assert len(df_row) == 1

                    x = np.array(Rs)
                    y = [df_row['recall@{}'.format(r)].values[0] for r in x]
                    ax.plot(x, y, label=algo)
                    ax.legend()

    # ------------------------ legend

    ax = axes.ravel()[-1]
    leg_lines, leg_labels = ax.get_legend_handles_labels()
    # for some reason, each algo appears 3x, so just take first
    leg_lines, leg_labels = leg_lines[:len(ALGOS)], leg_labels[:len(ALGOS)]

    plt.figlegend(leg_lines, leg_labels, loc='lower center',
                  ncol=len(ALGOS), labelspacing=0)

    # ------------------------ axis cleanup / formatting

    # configure all axes
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            title = "{}, {}B".format(DATASETS[i], NBYTES_LIST[j])
            ax.set_title(title, y=1.01)
            ax.set_ylim([0, 1])
            ax.set_xscale("log")

            # remove all legends except the very last one
            if (i != len(axes) or j != len(ax_row)) and ax.legend_:
                ax.legend_.remove()

    # remove x labels except for bottom axis
    for ax in axes[:-1, :].ravel():
        plt.setp(ax.get_xticklabels(), visible=False)
        # ax.get_xaxis().set_visible(False)

    if axes.shape[1] > 1:
        # hide y axis for axes not in left col
        for i, ax in enumerate(axes[:, 1:].ravel()):
            # pass
            # ax.get_yaxis().set_visible(False)
            ax.get_yaxis().set_ticklabels([], labelpad=-10, fontsize=1)

        # ylabel left col
        for i, ax in enumerate(axes[:, 0].ravel()):
            ax.set_ylabel("Recall@R")

        # xlabel bottom rows
        for i, ax in enumerate(axes[-1, :].ravel()):
            # no idea why we need the dummy tick at the beginning
            ax.set_xticklabels(['', '0', '1', '2', ''])
            # ax.set_xticklabels(['', '0', '1', '2', '3'])

        axes[-1, -1].set_xticklabels(['', '0', '1', '2', '3'])
        axes[-1, 1].set_xlabel("Log10(R)")

    # ------------------------ show / save plot

    # plt.tight_layout(h_pad=.02, w_pad=.02)
    plt.tight_layout(w_pad=.02)
    # plt.subplots_adjust(top=.88, bottom=.21, hspace=.4)
    plt.suptitle(suptitle, fontsize=16)
    plt.subplots_adjust(top=.91, bottom=.11)
    save_fig(fname)
    # plt.show()


def distortion_fig(fake_data=False, l2=True, suptitle=None, fname='l2_distortion'):
    # experiment params:
    #   datasets = Sift1M, Convnet1M, LabelMe22k, MNIST
    #   bytes = [8, 16, 32]
    # layout: [ndatasets x nums_bytes] (ie, [4x3])
    #   each subplot a barplot showing corr with err bars

    DATASETS = ['Sift1M', 'Convnet1M', 'LabelMe', 'MNIST']
    ALGOS = ['Bolt', 'PQ', 'OPQ', 'PairQ']
    NBYTES_LIST = [8, 16, 32]

    figsize = (6, 8)
    sb.set_context("talk", rc={'xtick.major.pad': 3})
    set_palette(ncolors=len(ALGOS))
    # fig, axes = plt.subplots(4, 3)
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.reshape((4, 1))

    if suptitle is None:
        suptitle = 'Quality of Approximate Distances'

    # fake_data = data is None
    if fake_data:
        algo2offset = {'Bolt': .4, 'PQ': .3, 'OPQ': .45, 'PairQ': .5}
        nfake_corrs = 10

        dicts = []
        for dataset in DATASETS:
            for nbytes in NBYTES_LIST:
                for algo in ALGOS:
                    if fake_data:
                        corrs = np.random.rand(nfake_corrs) / 2.
                        corrs += algo2offset[algo]
                        corrs *= .9 + .1 * nbytes / 32.
                    params = {'algo': algo, 'dataset': dataset,
                              'nbytes': '{}B'.format(nbytes)}
                    dicts += [dict(params, **{'corr': c}) for c in corrs]

        # data = pd.DataFrame.from_records(dicts, index=[0])
        data = pd.DataFrame.from_records(dicts)
        # print data
        # return

        # ------------------------ plot the data

        for d, dataset in enumerate(DATASETS):
            # df_dataset = data.loc[data['dataset'] == dataset]
            df = data.loc[data['dataset'] == dataset]
            df.rename(columns={'algo': ' '}, inplace=True)  # hide from legend

            ax = axes.ravel()[d]
            sb.barplot(x='nbytes', y='corr', hue=' ', data=df, ax=ax)

    else:
        DATASETS = ['Sift1M', 'Convnet1M', 'LabelMe', 'MNIST']
        # ALGOS = ['Bolt', 'PQ', 'OPQ', 'PairQ']
        # DATASETS = ['Convnet1M', 'MNIST']
        # ALGOS = ['PQ', 'OPQ']
        # ALGOS = ['PQ4', 'PQ', 'OPQ']
        ALGOS = ['Bolt No Quantize', 'PQ', 'OPQ']
        for d, dset in enumerate(DATASETS):
            if l2:
                path = os.path.join('../results/correlation_l2/', dset, 'all_results.csv')
            else:
                path = os.path.join('../results/correlation_dotprods/', dset, 'all_results.csv')

            df = pd.read_csv(path)

            print "path: ", path

            pq4 = (df['_algo'] == 'PQ') & (df['_code_bits'] == 4)
            df.loc[pq4, '_algo'] = 'Bolt No Quantize'
            bolt_rot = (df['_algo'] == 'Bolt') & (df['opq_iters'] > 0)
            df = df.loc[~bolt_rot]

            # print df.loc[df['_algo'] == 'PQ4']
            # print df.loc[df['_algo'] == 'PQ4']
            # return

            df.rename(columns={'_algo': ' '}, inplace=True)

            # df['nbytes'] = df['_code_bits'] * df['_ncodebooks'] / 8
            all_nbytes = (df['_code_bits'] * df['_ncodebooks'] / 8).values
            df['nbytes'] = ["{}B".format(b) for b in all_nbytes.astype(np.int)]

            ax = axes.ravel()[d]
            sb.barplot(x='nbytes', y='corr', hue=' ', data=df, ax=ax)

            ax.set_title(dset)

    # ------------------------ legend

    ax = axes.ravel()[-1]
    leg_lines, leg_labels = ax.get_legend_handles_labels()
    plt.figlegend(leg_lines, leg_labels, loc='lower center',
                  ncol=2, labelspacing=0)

    # ------------------------ axis cleanup / formatting

    # configure all axes
    for i, ax in enumerate(axes.ravel()):
        # title = "{}".format(DATASETS[i]) # TODO uncomment
        # ax.set_title(title, y=1.01) # TODO uncomment
        ax.set_ylim([0, 1])
        ax.set_xlabel('', labelpad=-10)
        if l2:
            ax.set_ylabel('Correlation With\nTrue Distance')
        else:
            ax.set_ylabel('Correlation With\nTrue Dot Product')
        if ax.legend_:
            ax.legend_.remove()

    # ------------------------ show / save plot

    # plt.tight_layout()  # for fig size 6x9
    plt.tight_layout(h_pad=.8)
    plt.suptitle(suptitle, fontsize=16)
    # plt.subplots_adjust(top=.92, bottom=.08)  # for fig size 6x9
    # plt.subplots_adjust(top=.90, bottom=.08)
    plt.subplots_adjust(top=.90, bottom=.1)
    save_fig(fname)
    # plt.show()


def kmeans_fig(data=None, fname='kmeans'):
    # bolt vs raw floats, k=16 on top and k=32 on the bottom

    ALGOS = ['Bolt', 'Matmul']
    Ks = [16, 64]

    sb.set_context("talk")
    set_palette()
    figsize = (6, 6)
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    fake_data = data is None
    if fake_data:
        dicts = []

        bolt_times = np.linspace(0, 100, 21)
        bolt_errs = np.max(Ks) * np.exp(-.1 * bolt_times)

        matmul_times = np.linspace(0, 100, 11)
        matmul_errs = np.max(Ks) * np.exp(-.05 * matmul_times)

        for i in range(3):  # simulate multiple trials
            # bolt_errs *= (1 + .2 * np.random.randn(*bolt_errs.shape))
            # matmul_errs *= (1 + .2 * np.random.randn(*matmul_errs.shape))
            bolt_errs += 5 * np.random.randn(*bolt_errs.shape)
            matmul_errs += 5 * np.random.randn(*matmul_errs.shape)
            bolt_errs = np.maximum(0, bolt_errs)
            matmul_errs = np.maximum(0, matmul_errs)
            bolt_errs = np.sort(bolt_errs)[::-1]
            matmul_errs = np.sort(matmul_errs)[::-1]
            for k in Ks:
                for t, err in zip(bolt_times, bolt_errs):
                    dicts.append({'trial': i, 'algo': 'Bolt', 'k': k, 't': t, 'err': err / k})
                for t, err in zip(matmul_times, matmul_errs):
                    dicts.append({'trial': i, 'algo': 'Matmul', 'k': k, 't': t, 'err': err / k})

        # data = pd.DataFrame.from_records(dicts, index=[0])
        data = pd.DataFrame.from_records(dicts)
        # print data
        # return

    # ------------------------ plot curves

    for i, k in enumerate(Ks):
        ax = axes[i]
        df = data.loc[data['k'] == k]
        df.rename(columns={'algo': ' '}, inplace=True)  # hide from legend

        # sb.tsplot(value='err', condition=' ', unit='k', time='t', data=df, ax=ax, n_boot=100)
        sb.tsplot(value='err', condition=' ', unit='trial', time='t', data=df,
                  ax=ax, ci=95, n_boot=500)

    # ------------------------ configure axes

    # configure all axes
    for i, ax in enumerate(axes.ravel()):
        title = "K-Means Convergence, K={}".format(Ks[i])
        ax.set_title(title, y=1.01)
        # ax.set_xlabel('', labelpad=-10)
        ax.set_xlabel('Wall Time (s)')
        # ax.set_ylabel('MSE')
        ax.set_ylabel('Mean Squared Error')

    axes[1].legend_.remove()

    # ------------------------ show / save plot

    plt.tight_layout()
    # plt.tight_layout(h_pad=.8)
    # plt.subplots_adjust(top=.92, bottom=.08)  # for fig size 6x9
    # plt.subplots_adjust(top=.90, bottom=.08)
    # save_fig(fname)
    plt.show()


def main():
    # pal = set_palette()
    # sb.palplot(pal)
    # plt.show()

    # popcount_fig()
    # encoding_fig()
    query_speed_fig(fname='query_speed_with_matmuls')
    # matmul_fig()
    # recall_r_fig(suptitle='Nearest Neighbor Recall', fname='l2_recall')
    # distortion_fig(fake_data=False, fname='l2_distortion')
    # distortion_fig(fake_data=False, fname='dotprod_distortion',
    #                suptitle='Quality of Approximate Dot Products', l2=False)
    # kmeans_fig()


if __name__ == '__main__':
    main()
