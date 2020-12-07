import time
import pickle
import sys
import time
import os
import bz2, lzma
import numpy as np
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
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
import tqdm
from tqdm.notebook import trange, tqdm
import glob
import itertools
import multiprocessing, functools, sys
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------------------------------------------------------------------------------------------------------------------------------------

def ncd_calc_file   (
                    file1,          file2,          compressor=lzma, \
                    c_level=9
                    ):

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

# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Medfilt func
def AC_medfilt(sample, medfilt_base=[3, 17]):
    s1 = scipy.signal.medfilt(sample,medfilt_base[0])
    n1 = scipy.signal.medfilt(s1,medfilt_base[1])
    s1 = s1 - n1
    s1 = s1/s1.max()
    ac_medfilt = s1
    return ac_medfilt
# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
def MB_process(sample, medfilt_base=[3, 17],   peak_cond=[6, 0.3],     ext=2):

    sample_mf = AC_medfilt(sample, medfilt_base=medfilt_base)
    sample_bin = ext_to_bin(sample_mf)
    return sample_bin
# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Medfilt Binary Normalized Compression Distance
def MB_NCD  (
            sample1,                sample2,                compressor='BZIP2', \
            medfilt_base=[3, 17],   peak_cond=[6, 0.3],     ext=2
            ):
    supported = ['BZIP2', 'LZMA']
    
    sample1_bin = MB_process(sample1, medfilt_base=medfilt_base, peak_cond=peak_cond, ext=ext)
    sample2_bin = MB_process(sample2, medfilt_base=medfilt_base, peak_cond=peak_cond, ext=ext)
    # sample1_mf = AC_medfilt(sample1, medfilt_base=medfilt_base)
    # sample2_mf = AC_medfilt(sample2, medfilt_base=medfilt_base)

    # sample1_bin = ext_to_bin(sample1_mf, peak_cond, ext)
    # sample2_bin = ext_to_bin(sample2_mf, peak_cond, ext)
    a = str(np.random.randint(0, 64))

    dir_path = os.path.dirname(os.path.realpath(__file__))

    sample1_file = dir_path + "/temps/1_" + str(os.getpid()) + "_" + a + f'{datetime.now():%H_%M_%S_%f}' + "_tuningTemp.csv"
    sample2_file = dir_path + "/temps/2_" + str(os.getpid()) + "_" + a + f'{datetime.now():%H_%M_%S_%f}' + "_tuningTemp.csv"

    np.savetxt(sample1_file, np.array(sample1_bin))
    np.savetxt(sample2_file, np.array(sample2_bin))
        

    if compressor in supported:
        if compressor == 'LZMA':
            compressor = lzma
            ncd = ncd_calc_file(sample1_file, sample2_file, compressor=compressor)
        elif compressor == 'BZIP2':
            compressor = bz2
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
# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Random sample from database
def randomize_sample(
                    recon_data,             current_kill_zone,      ss=-1,          \
                    trials_per_class=741,   class_num=7,            sensors_num=81, \
                    arr_shape=[9, 9],       indexes=[-1, -1, -1],   labels=[-1, -1] \
                    ):
    # Randomizing indexes and labels to test

    # Fixed indexes or random
    if indexes[0] == -1:
        idx1 = np.random.randint(0, trials_per_class)
    else:
        idx1 = indexes[0]

    if indexes[1] == -1:
        idx2 = np.random.randint(0, trials_per_class)
    else:
        idx2 = indexes[1]
        
    if indexes[2] == -1:
        idx3 = np.random.randint(0, trials_per_class)
    else:
        idx3 = indexes[2]

    # Fixed classes or random
    if labels[0] == -1:
        label1 = np.random.randint(1, class_num+1)
    else:
        label1 = labels[0]
    
    if labels[1] == -1:
        # Needs different labels
        choices = [x for x in range(1, class_num+1) if x != label1]
        label2 = np.random.choice(choices)
    else:
        label2 = labels[1]

    # Only use sensors NOT in killzone
    if ss == -1:
        choices = [x for x in range(sensors_num) if x not in current_kill_zone]
        sensor = np.random.choice(choices)
    else:
        sensor = ss
    # --------------------------------------

    sample1 = np.array(recon_data[label1])[idx1][:, sensor//arr_shape[1], sensor%arr_shape[0]]
    sample2 = np.array(recon_data[label1])[idx2][:, sensor//arr_shape[1], sensor%arr_shape[0]]
    sample3 = np.array(recon_data[label2])[idx3][:, sensor//arr_shape[1], sensor%arr_shape[0]]
    # --------------------------------------------------------------------

    return sample1, sample2, sample3, sensor
# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# MB_NCD distance test between samples
def run_test(
            sample1,                sample2,                sample3, 
            compressor='BZIP2',     medfilt_base=[3, 51],   peak_cond=[5, 0.5], \
            ext=2,                  verbose=0
            ):
    ncd_same = MB_NCD(sample1, sample2, compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond, ext=2)
    ncd_diff = MB_NCD(sample3, sample2, compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond, ext=2)
    if verbose:
        print(ncd_same, ncd_diff)
    if (ncd_same >= ncd_diff):
        return False, ncd_same, ncd_diff
    elif ncd_same < ncd_diff:
        # score_rec_bzip2[-1] = score_rec_bzip2[-1] + 1
        return True, ncd_same, ncd_diff
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

        
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Hypertuning made for multiprocessing
def ncd_param_tuning(recon_data, tune_set, rand_trial_num=183, wrongs=[], compressors='b', \
        killzone = [0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 27, 53, 62, 70, 71, 72, 77, 78, 79, 80], \
        tuning_output="tuning_output.txt", sensors_num=81):

    print("Worker %d is processing file %s\n" % (os.getpid(), str(tune_set)))

    conf_rec = {}
    score_rec_bzip2 = 0
    wrongs = [0 for x in range(sensors_num)]

    
    usable_configs = []
    usable_configs_sensors = []
    usable_configs_best = []
    total_trash = []

    [x, y, z, n] = tune_set
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
    tested = [0 for x in range(sensors_num)]
    miss = [0 for x in range(sensors_num)]

    # Random trials
    for g, positive_trial in enumerate(list(range(rand_trial_num))):

        if len(current_kill_zone) >= (sensors_num):
            error = "Kill zone for conf " + str(current_conf) + " has exceeded 81. Consider this bad config\n"
            print(error)
            sys.stdout.flush()
            # with open (tuning_output, 'a') as f:
            #     f.write(error)
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

            while keep_testing and (len(current_kill_zone) < (sensors_num)):
                # pbar.set_description(str(current_conf)+ ": " + str(81 - len(current_kill_zone)))
                test_result, _1, _2  = run_test(sample1, sample2, sample3, \
                    compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond, ext=2)
                ncd_fail = not test_result
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
                            if len(current_kill_zone) >= (sensors_num):
                                break
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

                    if len(current_kill_zone) >= (sensors_num):
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

            if len(current_kill_zone) >= (sensors_num):
                error = "Kill zone for conf " + str(current_conf) + " has exceeded 81. Consider this bad config\n"
                print(error)
                sys.stdout.flush()
                with open (tuning_output, 'a') as f:
                    f.write(error)
                usable = False
                                            
        if len(current_kill_zone) < 80:
            # pbar.update(1)
            pass
        else:
            # pbar.update(rand_trial_num - g)
            pass
            

    
    # if usable:
    #     usable_configs.append(current_conf) 
    #     good_sensors = [x for x in range(sensors_num) if x not in current_kill_zone]
    #     usable_configs_sensors.append(good_sensors)
        
    #     tested_clean = [tested[x] for x in good_sensors]
    #     miss_clean = [miss[x] for x in good_sensors]
    #     best_sensors = [x for x in good_sensors if miss[x] < 1]
    #     score = []

    #     usable_configs_best.append(best_sensors)
    #     # bzip2_score = float( (sum(tested_clean) - sum(miss_clean)) / float(sum(tested_clean)) )

    #     if len(best_sensors) > 0:
    #         to_write = "Bzip2 config: " + str(current_conf) + " good for " + \
    #             str(best_sensors) + " miss: " + str(sum(miss_clean)) + "/" + str(sum(tested_clean)) + '\n'
    #         with open (tuning_output, 'a') as f:
    #             f.write(to_write)
    #         print(to_write)
    #         sys.stdout.flush()                
 
    if usable:
        
        good_sensors = [x for x in range(sensors_num) if x not in current_kill_zone]
        print(current_conf, good_sensors)
        usable_configs = current_conf
        usable_configs_sensors = good_sensors
        tested_clean = [tested[x] for x in good_sensors]
        miss_clean = [miss[x] for x in good_sensors]
        best_sensors = [[x, miss[x], tested[x]] for x in good_sensors if miss[x] < 1]
        # best_sensors_score = [] for x in best_sensors]
        usable_configs_best = best_sensors
        if len(best_sensors) > 0:
            to_write = "Bzip2 config: " + str(current_conf) + " good for " + \
                str(best_sensors) + " miss: " + str(sum(miss_clean)) + "/" + str(sum(tested_clean)) + '\n'
            with open (tuning_output, 'a') as f:
                f.write(to_write)
            print(to_write)
            sys.stdout.flush()  
    return usable_configs, usable_configs_sensors, usable_configs_best
# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# select clustering
def pairwise_clustering(
            X_train, start_centers, cluster_func="pycls_kmeans", \
            X_test=None, cluster_param=[4], compressor='BZIP2', medfilt_base=[3, 21], peak_cond=[21, 0.6]):

    user_function = MB_NCD
    
    # for knearest cluster param = [k, ]

    if cluster_func == "pycls_kmeans":
        metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

        # create K-Means algorithm with specific distance metric
        
        kmeans_instance = kmeans(X_train, start_centers, metric=metric)

        # run cluster analysis and obtain results
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()

        centers = kmeans_instance.get_centers()

        print(clusters)

    # elif cluster_func == "sklrn_kmeans":

    #     mask = len(in_arr)//2

    #     zeores = [0 for x in range(mask)]
    #     ones = [1 for x in range(mask)]
    #     label = zeores + ones
    
    #     assert(len(label) == len(in_arr))
        
    #     knn = NearestNeighbors(n_neighbors=2, 
    #         algorithm='auto', 
    #         metric=lambda a,b: user_function(a, b, compressor=compressor, medfilt_base=medfilt_base, peak_cond=peak_cond))
    #     out = KMeans(n_clusters=2, random_state=0).fit(in_arr)
    #     # out = knn.kneighbors(test)

    #     assert(abs(out[0][-1][0] - MB_NCD(test[-1], in_arr[out[1][-1][0]], compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond)) < 0.000001)
    #     print(out)


    elif cluster_func == "sklrn_nneighbor":

        mask = len(X_train)//2

        zeores = [0 for x in range(mask)]
        ones = [1 for x in range(mask)]
        label = zeores + ones
    
        assert(len(label) == len(in_arr))
        
        knn = NearestNeighbors(n_neighbors=2, 
            algorithm='auto', 
            metric=lambda a,b: user_function(a, b, compressor=compressor, medfilt_base=medfilt_base, peak_cond=peak_cond))
        knn.fit(X_train)
        out = knn.kneighbors(X_test)

        assert(abs(out[0][-1][0] - MB_NCD(test[-1], X_train[out[1][-1][0]], compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond)) < 0.000001)
        print(out)


    elif cluster_func == "sklrn_kneighbor":

        mask = len(X_train)//2

        zeores = [0 for x in range(mask)]
        ones = [1 for x in range(mask)]
        y_train = zeores + ones
        # print(label)
        # print(X_train)
        assert(len(y_train) == len(y_train))
        
        y_test = y_train
        
        knn = KNeighborsClassifier(n_neighbors=cluster_param[0], 
            metric=lambda a,b: user_function(a, b, compressor=compressor, medfilt_base=medfilt_base, peak_cond=peak_cond))
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        # assert(abs(out[0][-1][0] - MB_NCD(test[-1], in_arr[out[1][-1][0]], compressor='BZIP2', medfilt_base=medfilt_base, peak_cond=peak_cond)) < 0.000001)
        # print(out)
    else:
        print("CLUSTER FUNCTION NOT SUPPORTED")
        exit(0)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Hypertuning easy
def hypertuning_medfilt (
                            tune_set, in_arr, to_file=True
                        ):
    MAX_NPROCESS = multiprocessing.cpu_count() // 2
    p = multiprocessing.Pool(MAX_NPROCESS)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    test_choices = list(itertools.product(*tune_set))
    # print("Testing permutations: ", len(test_choices))

    test_choices = [x for x in test_choices if x[2] < x[3]]
    func = functools.partial(ncd_param_tuning, in_arr)

    usable_configs, usable_configs_sensors, usable_configs_best = zip(*p.map(func, test_choices))

    usable_configs =         list(usable_configs)        
    usable_configs_best =    list(usable_configs_best)   
    usable_configs_sensors = list(usable_configs_sensors)

    usable_configs_best =    [x for i, x in enumerate(usable_configs_best   ) if len(usable_configs[i]) > 0]
    usable_configs_sensors = [x for i, x in enumerate(usable_configs_sensors) if len(usable_configs[i]) > 0]
    usable_configs =         [x for i, x in enumerate(usable_configs        ) if len(usable_configs[i]) > 0]

    usable_configs =         np.squeeze(np.array(usable_configs)        )
    usable_configs_best =    np.squeeze(np.array(usable_configs_best)   )
    usable_configs_sensors = np.squeeze(np.array(usable_configs_sensors))

    if to_file:
        np.savetxt("usable_configs.csv", usable_configs, delimiter=",")
        np.savetxt("usable_configs_sensors.csv", usable_configs_sensors, delimiter=",")
        np.savetxt("usable_configs_best.csv", usable_configs_best, delimiter=",")
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

