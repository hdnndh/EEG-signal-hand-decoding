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
import multiprocessing, functools
from mb_ncd_utils import *


MAX_NPROCESS = multiprocessing.cpu_count() // 2
p = multiprocessing.Pool(MAX_NPROCESS)

dir_path = os.path.dirname(os.path.realpath(__file__))




tuning_output = "tuning_output.txt"



killzone = [0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 27, 53, 62, 70, 71, 72, 77, 78, 79, 80]

with open("mesh_ME_db_128.pickle", "rb") as f:
    recon_data = pickle.load(f)

in_arr = copy.deepcopy(recon_data)
print(np.array(recon_data[1]).shape)

dim1 = np.array(recon_data[1]).shape[0]
dim2 = np.array(recon_data[1]).shape[1]

label1 = 1
label2 = 6
sensor = 4
mask = 30
mask2 = dim1 - mask

assert(sensor not in killzone)

with open (tuning_output, 'a') as f:
    f.write("\n------------------------------------------------------------\n")
rough_tune_set = [[5, 15, 25], [0.3, 0.5, 0.7], [3, 5, 7], [15, 61, 91]]
fine_tune_set = [[17, 19, 21, 23, 25], [0.3], [3], [69, 71, 73, 75, 77]]
weird_tune_set = [[5, 11, 15, 21], [0.3, 0.4, 0.5, 0.6], [3, 11, 21, 41], [61, 21, 41, 81]]


test_choices = list(itertools.product(*weird_tune_set))
print("Testing permutations: ", len(test_choices))

test_choices = [x for x in test_choices if x[2] < x[3]]
func = functools.partial(ncd_param_tuning, in_arr)

# r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))
# usable_configs, usable_configs_sensors, usable_configs_best
usable_configs, usable_configs_sensors, usable_configs_best = zip(*p.map(func, test_choices))

usable_configs =         list(usable_configs)        
usable_configs_best =    list(usable_configs_best)   
usable_configs_sensors = list(usable_configs_sensors)

usable_configs_best =    [x for i, x in enumerate(usable_configs_best   ) if len(usable_configs[i]) > 0]
usable_configs_sensors = [x for i, x in enumerate(usable_configs_sensors) if len(usable_configs[i]) > 0]
usable_configs =         [x for i, x in enumerate(usable_configs        ) if len(usable_configs[i]) > 0]

arr1 = np.empty(len(usable_configs), object)
arr1[:] = usable_configs

arr2 = np.empty(len(usable_configs_sensors), object)
arr2[:] = usable_configs_sensors

arr3 = np.empty(len(usable_configs_best), object)
arr3[:] = usable_configs_best

print(list(usable_configs))
print(list(usable_configs_best))
print(list(usable_configs_sensors))
# ncd_param_tuning(recon_data, tune_set=weird_tune_set, rand_trial_num=183)
try:
    np.save("usable_configs4.npy", arr1)
    np.save("usable_configs_sensors4.npy", arr2)
    np.save("usable_configs_best4.npy", arr3)
except:
    pass
try:
    np.savetxt("usable_configs4.csv", arr1, fmt='%s', delimiter=",")
    np.savetxt("usable_configs_sensors4.csv", arr2, fmt='%s', delimiter=",")
    np.savetxt("usable_configs_best4.csv", arr3, fmt='%s', delimiter=",")
except:
    pass
print("\n\n===================================FINALEE===================================")




# ==========================RESULTS==========================
# PEAKS AND VALLEYS
# ---- GENERAL BEST 
# -------- #1 [21, 0.5, 3, 15]
# ---- LZ77
#
#
# ---- BZIP
# ---- #1 [21, 0.3, 3, 61]
#
# 
# PEAKS ONLY
#
#
#
# VALEYS ONLY
#
#
#
#================================================================================================-------------------
