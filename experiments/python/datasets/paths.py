#!/usr/env/python

import os

DATASETS_DIR = os.path.expanduser("~/Desktop/datasets/")


def to_path(*args):
    return os.path.join(DATASETS_DIR, *args)


# straightforward datasets
MSRC_12 = to_path('MSRC-12', 'origData')
UCR = to_path('ucr/UCRArchive_2018')
UCR_INFO = to_path('ucr/DataSummary.csv')
UWAVE = to_path('uWave', 'extracted')
PAMAP = to_path('PAMAP_Dataset')
PAMAP2 = to_path('PAMAP2_Dataset')
WARD = to_path('WARD1.0')
UCI_GAS = to_path('uci-gas-sensor')

# ampds2
AMPD2_POWER = to_path('ampds2', 'electric')
AMPD2_GAS = to_path('ampds2', 'gas')
AMPD2_WEATHER = to_path('ampds2', 'weather')
AMPD2_WATER = to_path('ampds2', 'water')

# caltech-{101,256}
CALTECH_101 = to_path('caltech', '101_ObjectCategories')
CALTECH_256 = to_path('caltech', '256_ObjectCategories')

# ECG data
SHAREE_ECG = to_path('sharee-ecg-database')
INCART_ECG = to_path('incart-12-lead-ecg')
