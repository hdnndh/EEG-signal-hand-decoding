# MedianFilter_BinaryTransform  

AC_medfilt in util file was suggested and sample code provided by Professor Andrew Cohen (Drexel University)

Database taken from https://github.com/ductai199x/eeg-deeplearning

Although it was called Binary Transform, the process when extrema choice (ext) = 2 for most function described convert the signal into 1, 0 and -1 for peaks, between, and valleys.

Most of the helper functions are in mb_ncd_utils.py

usable_configs*.csv contains the config permutation tested

usable_configs_best*.csv contains corresponding no-error compatible sensors and their score in format \[sensor, 0, score\]; 0 is just for assertion/validation

usable_configs_sensors*.csv contains corresponding compatible sensors with less than 3 trial errors. 
