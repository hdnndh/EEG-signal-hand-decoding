import time
import pickle
import sys
import time
import os
import bz2, lzma
import numpy as np
from ncdlib import compute_ncd, available_compressors
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.signal
from sklearn.decomposition import FastICA
import pandas as pd
import copy
from numpy import NaN
from datetime import datetime
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.utils.metric import type_metric, distance_metric
from sklearn.neighbors import NearestNeighbors
import tqdm
from tqdm.notebook import trange, tqdm
import glob
import itertools
import multiprocessing, functools, sys
from mb_ncd_utils import *
import scipy
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from sklearn.metrics import classification_report, confusion_matrix

# Data and variable init____________________________________________________________________________________________________
killzone = [0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 27, 53, 62, 70, 71, 72, 77, 78, 79, 80]

with open("_mesh_ME_db_128.pickle", "rb") as f:
    recon_data = pickle.load(f)

in_arr = copy.deepcopy(recon_data)
print(np.array(recon_data[1]).shape)

dim1 = np.array(recon_data[1]).shape[0]
dim2 = np.array(recon_data[1]).shape[1]


# Configs_____________________________________________________________________________________________________________________
# Bzip2 config: [21, 0.6, 3, 21] good for [61] miss: 0/5
# Bzip2 config: [11, 0.3, 21, 81] good for [41] miss: 0/2
# Best so far [11, 0.3, 21, 81] #41

# filters
medfilt_featfilt = 3
medfilt_basefilt = 41

# Peaks condition
peak_prominence = 0.6
refractory_time = 15

# Sensor that works well with above configs
# 22/29
sensor = 14

assert(sensor not in killzone)

sample1, sample2, sample3, _ = randomize_sample(
    recon_data=recon_data, current_kill_zone=killzone, ss=sensor
    )

print(run_test(sample1, sample2, sample3, compressor='BZIP2', medfilt_base=[medfilt_featfilt, medfilt_basefilt], peak_cond=[refractory_time, peak_prominence], ext=2, verbose=1))
# corr = scipy.signal.correlate(sample1, sample2)

# ===============================================================================================================================
# Setup for clustering



# number of samples per label
mask = 10
limit = mask * 2
label1 = 2
label2 = 4


# EEG dat to test
eeg_dat = []
for idx in range(mask):
    sample = np.array(recon_data[label1])[idx][:, sensor//9, sensor%9]
    eeg_dat.append(sample)

for idx in range(mask):
    sample = np.array(recon_data[label2])[idx][:, sensor//9, sensor%9]
    eeg_dat.append(sample)


eeg_test = []
for idx in range(mask, limit):
    sample = np.array(recon_data[label1])[idx][:, sensor//9, sensor%9]
    eeg_test.append(sample)

for idx in range(mask, limit):
    sample = np.array(recon_data[label2])[idx][:, sensor//9, sensor%9]
    eeg_test.append(sample)


print("TRAIN DATA", "from", 0, "to", mask)
print(np.array(eeg_dat).shape)

print("TEST DATA", "from", mask, "to", limit)
print(np.array(eeg_test).shape)


start_centers = [eeg_dat[0], eeg_dat[mask+1]]
sample = eeg_dat


pairwise_clustering(eeg_dat, start_centers, cluster_func="sklrn_kneighbor", X_test=eeg_test, compressor='BZIP2', medfilt_base=[medfilt_featfilt, medfilt_basefilt], peak_cond=[refractory_time, peak_prominence])
# sel_clustering(eeg_dat, start_centers, cluster_func="pycls_kmeans", test=eeg_test)