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
from tqdm import trange, tqdm
import glob

dir_path = os.path.dirname(os.path.realpath(__file__))

def ncd_calc_file(file1, file2, compressor=lzma, c_level=9):

    with open(file1, 'rb') as f:
        signal1_byte = f.read()

    with open(file2, 'rb') as f:
        signal2_byte = f.read()


    signal_concat = signal1_byte + signal2_byte


    if compressor is lzma:
        signal1_compressed = compressor.compress(signal1_byte, preset=c_level)
        signal2_compressed = compressor.compress(signal2_byte, preset=c_level)
        concat_compressed = compressor.compress(signal_concat, preset=c_level)
    else:
        signal1_compressed = compressor.compress(signal1_byte, c_level)
        signal2_compressed = compressor.compress(signal2_byte, c_level)
        concat_compressed = compressor.compress(signal_concat, c_level)

    
    n = len(concat_compressed) - min(len(signal1_compressed), len(signal2_compressed))
    d = max(len(signal1_compressed), len(signal2_compressed))

    ncd = n/float(d)

    return ncd


# ====================================================================================


# Medfilt func
def AC_medfilt(sample, medfilt_base=[3, 17]):
    s1 = scipy.signal.medfilt(sample,medfilt_base[0])
    n1 = scipy.signal.medfilt(s1,medfilt_base[1])
    s1 = s1 - n1
    s1 = s1/s1.max()
    ac_medfilt = s1
    return ac_medfilt


# Binary conversion
def ext_to_bin(in_sig, peak_cond=[5, 0.5], ext=2):
    sample_bin = copy.deepcopy(in_sig)

    peaks, _ = scipy.signal.find_peaks(in_sig, distance=peak_cond[0], prominence=peak_cond[1])
    valleys, _ = scipy.signal.find_peaks(-in_sig, distance=peak_cond[0], prominence=peak_cond[1])

    if ext == 2:
        extremas = np.append(peaks, valleys)
        sample_bin[peaks] = 1
        sample_bin[valleys] = -1
    elif ext == 1:
        extremas = peaks
        sample_bin[peaks] = 1
    elif ext == 0:
        extremas = valleys
        sample_bin[valleys] = -1
    else:
        extremas = np.append(peaks, valleys)
        sample_bin[peaks] = 1
        sample_bin[valleys] = -1
    
    for i, x in enumerate(sample_bin):
        if i not in extremas:
            sample_bin[i] = 0

    return sample_bin


# Medfilt Binary Normalized Compression Distance
def MB_NCD (sample1, sample2, compressor='LZ77', medfilt_base=[3, 17], peak_cond=[6, 0.3], ext=2):
    other_supported = ['LZMA']
    
    sample1_mf = AC_medfilt(sample1, medfilt_base=medfilt_base)
    sample2_mf = AC_medfilt(sample2, medfilt_base=medfilt_base)

    sample1_bin = ext_to_bin(sample1_mf, peak_cond, ext)
    sample2_bin = ext_to_bin(sample2_mf, peak_cond, ext)
    a = str(np.random.randint(0, 8))

    dir_path = os.path.dirname(os.path.realpath(__file__))

    sample1_file = dir_path + "/temps/1_" + a + f'{datetime.now():%H_%M_%S_%f}' + "_tuningTemp.csv"
    sample2_file = dir_path + "/temps/2_" + a + f'{datetime.now():%H_%M_%S_%f}' + "_tuningTemp.csv"

    np.savetxt(sample1_file, np.array(sample1_bin))
    np.savetxt(sample2_file, np.array(sample2_bin))
        

    if compressor in available_compressors():
        ncd = compute_ncd(sample1_file, sample2_file, compressor=compressor)
        # fileList = glob.glob(dir_path + '/*_tuningTemp.csv')
        
    elif compressor in other_supported:
        if compressor == 'LZMA':
            compressor = lzma
            ncd = ncd_calc_file(sample1_file, sample2_file, compressor=compressor)
        else:
            print("COMPRESSOR NOT SUPPORTED!!")
            exit(0)
    else:
        print("COMPRESSOR NOT SUPPORTED!!")
        exit(0)

    os.remove(sample1_file)
    os.remove(sample2_file)

    return ncd


tuning_output = "tuning_output2.txt"


def randomize_sample(recon_data, current_kill_zone, ss=-1, trials_per_class=741, class_num=7):
    # Randomizing indexes and labels to test
    idx1 = np.random.randint(0, trials_per_class)
    idx2 = np.random.randint(0, trials_per_class)
    idx3 = np.random.randint(0, trials_per_class)
    label1 = np.random.randint(1, class_num+1)

    # Needs different labels
    choices = [x for x in range(1, class_num+1) if x != label1]
    label2 = np.random.choice(choices)

    # Only use sensors NOT in killzone
    choices = [x for x in range(81) if x not in current_kill_zone]
    if ss == -1:
        sensor = np.random.choice(choices)
    else:
        sensor = ss
    # --------------------------------------

    sample1 = np.array(recon_data[label1])[idx1][:, sensor//9, sensor%9]
    sample2 = np.array(recon_data[label1])[idx2][:, sensor//9, sensor%9]
    sample3 = np.array(recon_data[label2])[idx3][:, sensor//9, sensor%9]
    # --------------------------------------------------------------------

    return sample1, sample2, sample3, sensor

def run_test(sample1, sample2, sample3, compressor='BZIP2', medfilt_base=[3, 51], peak_cond=[5, 0.5], ext=2):
    ncd_same = MB_NCD(sample1, sample2, compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond, ext=2)
    ncd_diff = MB_NCD(sample3, sample2, compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond, ext=2)
    if (ncd_same >= ncd_diff):
        return True
    elif ncd_same < ncd_diff:
        # score_rec_bzip2[-1] = score_rec_bzip2[-1] + 1
        return False

def ncd_param_tuning(recon_data, tune_set, rand_trial_num=128, wrongs=[], compressors='b'):
    conf_rec = {}
    score_rec_bzip2 = 0
    wrongs = [0 for x in range(81)]
    total_trial_num = 1
    for x in tune_set:
        total_trial_num *= len(x)
    total_trial_num *= rand_trial_num
    print("Trial count", total_trial_num)
    
    usable_configs = []
    usable_configs_sensors = []
    usable_configs_best = []
    total_trash = []
    
    with tqdm(total=total_trial_num) as pbar:
        for i, x in enumerate(tune_set[0]):
            for j, y in enumerate(tune_set[1]):
                for k, z in enumerate(tune_set[2]):
                    for m, n in enumerate(tune_set[3]):
                        current_kill_zone = copy.deepcopy(killzone)

                        # Setup for MB_NCD
                        peak_cond = [x, y]
                        medfilt_base = [z, n]
                        current_conf = [x, y, z, n]

                        # Record sensors that work well with current filter config
                        bad_sensors = []
                        bad_sensors_trigger = []
                        usable = True
                        
                        bzip2_score = 0
                        tested = [0 for x in range(81)]
                        miss = [0 for x in range(81)]

                        # Random trials
                        for g, positive_trial in enumerate(list(range(rand_trial_num))):

                            if len(current_kill_zone) >= 80:
                                error = "Kill zone for conf " + str(current_conf) + " has exceeded 80. Consider this bad config\n"
                                print(error)
                                with open (tuning_output, 'a') as f:
                                    f.write(error)
                                usable = False
                                total_trash.append(current_conf)
                                break
                            
                            # if killzone hasn't been filled, randomize data
                            sample1, sample2, sample3, sensor = randomize_sample(recon_data, current_kill_zone)

                            # compressors test string
                            # ---- b for bzip2
                            # ---- 7 for lz77
                            # ---- l for lzma
                            # ---- Ex: bl7
                            if compressors.find('b') != -1:
                                keep_testing = True
                                repeating_trial = False
                                # print("New")

                                while keep_testing and (len(current_kill_zone) < 80):
                                    pbar.set_description(str(current_conf)+ ": " + str(81 - len(current_kill_zone)))
                                    ncd_fail = run_test(sample1, sample2, sample3, \
                                        compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond, ext=2)
                                    tested[sensor] += 1
                                    # print("while status", keep_testing, len(current_kill_zone))
                                    # if ncd fail to work
                                    if ncd_fail:
                                        miss[sensor] += 1
                                        # if this is the first time current config fail on this sensor
                                        if sensor not in bad_sensors:
                                            # print("bad sensor encountered", sensor)
                                            # Record the failed attempt
                                            bad_sensors.append(sensor)
                                            bad_sensors_trigger.append(1)
                                            sample1, sample2, sample3, sensor = randomize_sample(recon_data, current_kill_zone, ss=sensor)

                                            # press the current config on the sensor have wrong trial again
                                            keep_testing = True
                                            repeating_trial = True

                                        # This is the 2nd+ time current config fail on this sensor
                                        else:
                                            # print("bad sensor encountered again", sensor)
                                            # Increment record
                                            bad_sensors_trigger[bad_sensors.index(sensor)] += 1
                                            # 3 times fail = byebye
                                            if bad_sensors_trigger[bad_sensors.index(sensor)] > 3:
                                                # print("kill zone updated", sensor)
                                                current_kill_zone.append(sensor)
                                                sample1, sample2, sample3, sensor = randomize_sample(recon_data, current_kill_zone)
                                                
                                                # moveon to next test iteration in the outer loop
                                                keep_testing = False

                                            # test the current sensor again in next iteration.
                                            # keep testing till fail or pass at least once
                                            else:
                                                # print("repeating trials")
                                                # print("Keep on keeping on: ", sensor, end='...', flush=True)
                                                sample1, sample2, sample3, sensor = randomize_sample(recon_data, current_kill_zone, ss=sensor)

                                                # Looping condition update
                                                keep_testing = True
                                                repeating_trial = True

                                        if len(current_kill_zone) >= 80:
                                            break

                                    elif repeating_trial == True:
                                        # print("Passed last iter, current status", tested[sensor], miss[sensor])
                                        if tested[sensor] - miss[sensor] > 3:
                                            keep_testing = False
                                            repeating_trial = False
                                            sample1, sample2, sample3, sensor = randomize_sample(recon_data, current_kill_zone)
                                        else:
                                            keep_testing = True
                                            repeating_trial = True
                                            sample1, sample2, sample3, sensor = randomize_sample(recon_data, current_kill_zone, ss=sensor)
                                    else:
                                        keep_testing = False
                                        repeating_trial = False
                                                                
                            if len(current_kill_zone) < 80:
                                pbar.update(1)
                                pass
                            else:
                                pbar.update(rand_trial_num - g)
                                

                        
                        if usable:
                            usable_configs.append(current_conf) 
                            good_sensors = [x for x in range(81) if x not in current_kill_zone]
                            usable_configs_sensors.append(good_sensors)
                            
                            tested_clean = [tested[x] for x in good_sensors]
                            miss_clean = [miss[x] for x in good_sensors]
                            best_sensors = [x for x in good_sensors if miss[x] <= 1]

                            usable_configs_best.append(best_sensors)
                            # bzip2_score = float( (sum(tested_clean) - sum(miss_clean)) / float(sum(tested_clean)) )

                            if len(best_sensors) > 0:
                                to_write = "Bzip2 config: " + str(current_conf) + " good for " + \
                                    str(best_sensors) + " miss: " + str(sum(miss_clean)) + "/" + str(sum(tested_clean)) + '\n'
                                with open (tuning_output, 'a') as f:
                                    f.write(to_write)

                            # score2 = ''
                                
                            # if bzip2_score > score_rec_bzip2:
                            #     score_rec_bzip2 = bzip2_score
                            #     conf_rec['BZIP2'] = current_conf
                            #     score2 = "BZIP2 score: " + str(score_rec_bzip2) + "/" + \
                            #         str(rand_trial_num) + " - Config: " + str(conf_rec['BZIP2']) + '\n'
                            
                            # with open (tuning_output, 'a') as f:
                            #     f.write(score2)
 
    np.savetxt("usable_configs", usable_configs)
    np.savetxt("usable_configs_sensors", usable_configs_sensors)
    np.savetxt("usable_configs_best", usable_configs_best)
    # score1 = "LZ77  final score: " + str(score_rec_lz77 ) + "/" + \
    #     str(rand_trial_num) + " - Config: " + str(conf_rec['LZ77'] ) + '\n'
    # with open (tuning_output, 'a') as f:
    #     f.write(score1)
    # score2 = "BZIP2 final score: " + str(score_rec_bzip2) + "/" + \
    #     str(rand_trial_num) + " - Config: " + str(conf_rec['BZIP2']) + '\n'
    # with open (tuning_output, 'a') as f:
    #     f.write(score2)
    # score3 = "LZMA final score: " + str(score_rec_lzma) + "/" + \
    #     str(rand_trial_num) + " - Config: " + str(conf_rec['LZMA']) + '\n'
    # with open (tuning_output, 'a') as f:
    #     f.write(score3)

    return usable_configs, usable_configs_sensors

killzone = [0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 27, 53, 62, 70, 71, 72, 77, 78, 79, 80]

with open("_mesh_ME_db_128.pickle", "rb") as f:
    recon_data = pickle.load(f)

print(np.array(recon_data[1]).shape)

dim1 = np.array(recon_data[1]).shape[0]
dim2 = np.array(recon_data[1]).shape[1]

label1 = 1
label2 = 6
sensor = 4
mask = 30
mask2 = dim1 - mask

assert(sensor not in killzone)

# %% TUNING CELL
# conf_rec = []
# score_rec_lz77 = []
# score_rec_bzip2 = []
with open (tuning_output, 'a') as f:
    f.write("\n------------------------------------------------------------\n")
rough_tune_set = [[5, 15, 25], [0.3, 0.5, 0.7], [3, 5, 7], [15, 61, 91]]
fine_tune_set = [[17, 19, 21, 23, 25], [0.3], [3], [69, 71, 73, 75, 77]]
weird_tune_set = [[5, 11, 15, 21], [0.3, 0.4, 0.5, 0.6], [3, 11, 21, 41], [61, 21, 41, 81]]
ncd_param_tuning(recon_data, tune_set=weird_tune_set, rand_trial_num=183)
# print(w)
# worst_sensor_idx = w.index(max(w))
# for x in killzone:
#     assert(w[x] == 0)
# with open (tuning_output, 'a') as f:
#     f.write("Worst signals came from sensor: " + str(worst_sensor_idx) + "\n")



                    
                    

print("\n\n===================================FINALEE===================================")
# best_conf_idx = score_rec_lz77.index(max(score_rec_lz77))
# best_conf = conf_rec[best_conf_idx]
# print("LZ77 score", max(score_rec_lz77), best_conf)

# best_conf_idx = score_rec_bzip2.index(max(score_rec_bzip2))
# best_conf = conf_rec[best_conf_idx]
# print("BZIP2 score", max(score_rec_bzip2), best_conf)



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


