#!/usr/bin/env python

import os
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from joblib import Memory

from . import paths
from . import files

_memory = Memory('./')


def _list_csvs(directory):
    return files.list_files(directory, endswith='.csv', abs_paths=True)


ELECTRIC_PATHS = _list_csvs(paths.AMPD2_POWER)
GAS_PATHS = _list_csvs(paths.AMPD2_GAS)
WATER_PATHS = _list_csvs(paths.AMPD2_WATER)
WEATHER_PATHS = _list_csvs(paths.AMPD2_WEATHER)

ELECTRIC_COLS = 'UNIX_TS,WHE,RSE,GRE,MHE,B1E,BME,CWE,DWE,EQE,FRE,HPE,OFE,' \
    'UTE,WOE,B2E,CDE,DNE,EBE,FGE,HTE,OUE,TVE,UNE'.split(',')

ELECTRIC_DATA_COLS = ELECTRIC_COLS[1:]
# ELECTRIC_DATA_COLS.remove('MHE')  # linear combo of other cols
# ELECTRIC_DATA_COLS.remove('UNE')  # linear combo of other cols
GAS_DATA_COLS = ['counter', 'avg_rate', 'inst_rate']
WATER_DATA_COLS = ['counter', 'avg_rate']


WEATHER_TIME_COL = 'Date/Time'
WEATHER_DATA_COLS = ['Temp (C)', 'Dew Point Temp (C)', 'Rel Hum (%)',
                     'Wind Dir (10s deg)', 'Wind Spd (km/h)',
                     'Visibility (km)', 'Stn Press (kPa)']
WEATHER_ALL_COLS = [WEATHER_TIME_COL] + WEATHER_DATA_COLS

FIG_SAVE_DIR = os.path.join('figs', 'ampds')


# ================================================================ public

class HouseRecording(object):

    def __init__(self, path, cols=None):
        data = _read_file(path)
        self.path = path
        self.name = os.path.basename(path).split('.')[0]
        self.col_names = cols
        self.sampleTimes = data[:, 0]
        self.data = data[:, 1:]  # XXX have to use all cols after the first

        # if 'power' in self.name:
        #     print "initial sample times: ", self.sampleTimes[:50]
        #     print

        # hack to deal with DWW water not having inst_rate
        # self.col_names = self.col_names[:self.data.shape[1]]
        self.data = self.data[:, :len(self.col_names)]


class WeatherRecording(object):

    def __init__(self):
        df = _load_weather_data()
        self.name = 'weather'
        self.col_names = WEATHER_DATA_COLS
        self.sampleTimes = _datetime_strs_to_unix_timestamps(df[WEATHER_TIME_COL])
        self.data = df[WEATHER_DATA_COLS].values.astype(np.float32)


# ------------------------ top-level data loading functions

def all_power_recordings():
    return [HouseRecording(path, cols=ELECTRIC_DATA_COLS) for path in ELECTRIC_PATHS]


def all_gas_recordings():
    return [HouseRecording(path, cols=GAS_DATA_COLS) for path in GAS_PATHS]


def all_water_recordings():
    return [HouseRecording(path, cols=WATER_DATA_COLS) for path in WATER_PATHS]


def all_weather_recordings():
    return [WeatherRecording()]  # just one data file, so just one recording


def all_timestamp_recordings():
    all_recordings = all_power_recordings() + all_gas_recordings() + \
        all_water_recordings() + all_weather_recordings()
    # all_recordings = all_weather_recordings() # TODO rm
    for r in all_recordings:
        r.data = r.sampleTimes.astype(np.float64)
        r.name += '_timestamps'

    return all_recordings


# ================================================================ private

# def _read_file(path, cols=None):
@_memory.cache
def _read_file(path):
    df = pd.read_csv(path).fillna(method='backfill')  # hold prev val
    # if cols is not None and len(cols) > 0:
    #     timestamps = df[df.columns[0]]
    # return df.values.astype(np.int32)
    return df.values.astype(np.float64)  # need f64 to not lose timestamps


@_memory.cache
def _load_weather_data():
    path = WEATHER_PATHS[0]
    df = pd.read_csv(path, sep=',').fillna(method='backfill')  # hold prev val
    return df[WEATHER_ALL_COLS]


def _datetimes_to_unix_timestamps(datetimes):
    # https://stackoverflow.com/q/34038273
    return (datetimes.astype(np.int64) / 1e6).astype(np.uint64)


def _datetime_strs_to_unix_timestamps(strs):
    return _datetimes_to_unix_timestamps(pd.to_datetime(strs))


# ================================================================ main

def save_fig_png(path):
    plt.savefig(path, dpi=300, bbox_inches='tight')


def _prev_corrs_stats(corr):
    assert corr.shape[0] == corr.shape[1]  # needs to be a correlation mat
    abs_corr = np.abs(corr)

    prev_corrs = np.zeros(len(corr) - 1)
    best_corrs = np.zeros(len(corr) - 1)
    for i, row in enumerate(abs_corr[1:]):  # each row after the first
        prev_corrs[i] = row[i]  # note that i is row index - 1
        try:
            best_corr_idx = np.nanargmax(row[:i+1])
            best_corrs[i] = row[best_corr_idx]
        except ValueError:  # if row all nans
            best_corrs[i] = prev_corrs[i]

        assert not (best_corrs[i] < prev_corrs[i])  # double neg for nans

    # avg corr with prev variable, avg highest corr with any preceding variable
    return np.nanmean(prev_corrs), np.nanmean(best_corrs)


def _plot_corr(data, fig, ax, add_title=True):
    """assumes data is row-major; ie, each col is one variable over time"""
    # cov = np.cov(data.T)
    corr = np.corrcoef(data.T)
    # im = ax.imshow(corr, interpolation='nearest',
    #                cmap=plt.cm.RdBu,
    #                norm=mpl.colors.Normalize(vmin=-1., vmax=1.))
    # fig.colorbar(im, ax=ax)
    # sb.heatmap(corr, center=0, ax=ax, square=True)
    sb.heatmap(corr, vmin=-1, vmax=1, center=0, ax=ax, square=True)

    if add_title:
        mean_prev_corr, mean_best_corr = _prev_corrs_stats(corr)
        ax.set_title("|rho| prev, best prev =\n{:.2f}, {:.2f}".format(
            mean_prev_corr, mean_best_corr))


def plot_recordings(recordings, interval_len=1000, norm_means=False,
                    mins_zero=False, savedir=None):

    for r in recordings:
        print(("recording {} has data of shape {}".format(r.name, r.data.shape)))
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))

        start_idxs = [0, len(r.data) - interval_len]
        end_idxs = [interval_len, len(r.data)]

        # any_nans_in_row = np.isnan(r.data).sum(axis=1)
        # print np.where(any_nans_in_row)[0]
        # continue

        for i, (start, end) in enumerate(zip(start_idxs, end_idxs)):
            timestamps = r.sampleTimes[start:end]
            data = r.data[start:end]
            if norm_means:
                data -= np.mean(data, axis=0).astype(data.dtype)
            elif mins_zero:
                data -= np.min(data, axis=0).astype(data.dtype)

            # print "data shape", data.shape
            # print "data final vals", data[-20:]
            # continue

            col = i + 1
            axes[0, col].plot(timestamps, data, lw=1)
            axes[1, col].plot(timestamps[1:], np.diff(data, axis=0), lw=1)
            axes[0, col].set_title('data')
            axes[1, col].set_title('first derivs')

        # plot correlation matrices for orig data and first derivs
        cor_sample_length = max(10000, len(r.data) // 5)
        data = r.data[:cor_sample_length]
        _plot_corr(data, fig, axes[0, 0])
        _plot_corr(np.diff(data, axis=0), fig, axes[1, 0])
        data = r.data[-cor_sample_length:]
        _plot_corr(data, fig, axes[0, -1])
        _plot_corr(np.diff(data, axis=0), fig, axes[1, -1])

        # _plot_corr(r.data[:cor_sample_length], fig, axes[0, 0])
        # data = r.data[-cor_sample_length:]
        # _plot_corr(data, fig, axes[2, 1])

        plt.tight_layout()
        # plt.show()

        if savedir is not None:
            files.ensure_dir_exists(savedir)
            # plt.savefig(os.path.join(savedir, r.name))
            save_fig_png(os.path.join(savedir, r.name))


def main():
    recordings = []
    recordings += all_gas_recordings()
    recordings += all_water_recordings()
    recordings += all_power_recordings()
    recordings += all_weather_recordings()

    norm_means = False
    # norm_means = True
    mins_zero = True

    plot_recordings(recordings, norm_means=norm_means, mins_zero=mins_zero,
                    savedir=FIG_SAVE_DIR)
    # plt.show()


if __name__ == '__main__':
    main()
