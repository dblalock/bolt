#!/usr/env/python

import os

DATASETS_DIR = os.path.expanduser("~/Desktop/datasets/")


def to_path(*args):
    return os.path.join(DATASETS_DIR, *args)


# straightforward datasets
MSRC_12 = to_path('MSRC-12', 'origData')
UCR_ORIG = to_path('ucr_data')
UCR = to_path('UCR_TS_Archive_2015')
UWAVE = to_path('uWave', 'extracted')
PAMAP = to_path('PAMAP_Dataset')
PAMAP2 = to_path('PAMAP2_Dataset')
WARD = to_path('WARD1.0')
UCI_GAS = to_path('uci-gas-sensor')

# ampds2
AMPD2_POWER = to_path('ampds2', 'electric')
# AMPD2_POWER = to_path('ampds2', 'electric', 'debug_all_power.csv')) # TOD
AMPD2_GAS = to_path('ampds2', 'gas')
AMPD2_WEATHER = to_path('ampds2', 'weather')
AMPD2_WATER = to_path('ampds2', 'water')

# ampds
DISHWASHER = to_path('AMPds', 'dishwasher_nohead.csv')
DISHWASHER_SHORT = to_path('AMPds', 'dishwasher_nohead_short.csv')
DISHWASHER_20K = to_path('AMPds', 'dishwasher_nohead_20k.csv')
DISHWASHER_LABELS = 'python/datasets/dishwasher-labels.txt'  # in project dir
DISHWASHER_LABELS_ALT = 'ts/python/datasets/dishwasher-labels.txt'  # proj dir


# TIDIGITS; the executable can be compiled from the source code here:
# https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools
# We just happen to have placed it in our tidigits subdirectory for
# convenience; it is not included with TIDIGITS.
TIDIGITS = to_path('tidigits', 'data')
SPH2PIPE_EXECUTABLE = to_path('tidigits', 'sph2pipe_v2.5', 'sph2pipe')


# compression benchmark
# COMPRESSION_DATASETS_DIR = to_path('compress')
COMPRESSION_ROWMAJOR_DIR = to_path('compress', 'rowmajor')
COMPRESSION_COLMAJOR_DIR = to_path('compress', 'colmajor')


# caltech-{101,256}
CALTECH_101 = to_path('caltech', '101_ObjectCategories')
CALTECH_256 = to_path('caltech', '256_ObjectCategories')

# UCD Database (just holter ECG signals)
# UCD_ECG = to_path('ucddb', 'ecg')

# SHAREE Database
SHAREE_ECG = to_path('sharee-ecg-database')

