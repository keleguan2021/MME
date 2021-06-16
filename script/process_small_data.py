"""
@Time    : 2021/6/14 20:06
@File    : process_small_data.py
@Software: PyCharm
@Desc    : 
"""
import os

import wfdb
import numpy as np
import pandas as pd
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal
from tqdm.std import tqdm
from sklearn.preprocessing import StandardScaler

MIT_ARRHYTHMIA_PATH = './data/mit-bih-arrhythmia-database-1.0.0'
MIT_ATRIAL_PATH = './data/mit-bih-atrial-fibrillation-database-1.0.0'

ATRIAL_CLASS_TO_NUM = {"AFIB": 0, "AFL": 1, "J": 2, "N": 3}

N = {"N", "L", "R"}
S = {"a", "J", "A", "S", "j", "e"}
V = {"V", "E"}
F = {"F"}
Q = {"/", "f", "Q"}

# All beats
BEATS = N.union(S, V, F, Q)
# Anomalous beats
ANOMALOUS_BEATS = S.union(V, F, Q)


def process_mit_arrhythmia(data_path):
    record_ids = list(map(lambda x: x.split('.')[0], list(filter(lambda x: x.endswith('.dat'), os.listdir(data_path)))))

    for idx in tqdm(record_ids):
        record = wfdb.rdrecord(os.path.join(data_path, idx))
        annotation = wfdb.rdann(os.path.join(data_path, idx), 'atr')

        signal_ch1 = record.p_signal[:, 0]
        signal_ch2 = record.p_signal[:, 0]

        ecg_ch1 = ecg.ecg(signal=signal_ch1, sampling_rate=record.fs, show=False)
        ecg_ch2 = ecg.ecg(signal=signal_ch2, sampling_rate=record.fs, show=False)

        # Smooth signals
        signal_smoothed_ch1 = ecg_ch1['filtered']
        signal_smoothed_ch2 = ecg_ch2['filtered']

        # Reading r-peaks
        r_peaks = ecg_ch1['rpeaks']

        # Reading annotations. `symbol` and `sample` are labels and values respectively.
        ann_symbol = annotation.symbol
        ann_sample = annotation.sample

        print(signal_ch1.shape, ann_sample, ann_symbol, r_peaks)


def process_mit_atrial(data_path):
    record_ids = list(map(lambda x: x.split('.')[0], list(filter(lambda x: x.endswith('.dat'), os.listdir(data_path)))))

    all_signals = []
    all_labels = []

    for idx in tqdm(record_ids):
        record = wfdb.rdrecord(os.path.join(data_path, idx))
        annotation = wfdb.rdann(os.path.join(data_path, idx), 'atr')

        signal_ch1 = record.p_signal[:, 0]
        signal_ch2 = record.p_signal[:, 0]

        print(signal_ch1.shape)

        ecg_ch1 = ecg.ecg(signal=signal_ch1, sampling_rate=record.fs, show=False)
        ecg_ch2 = ecg.ecg(signal=signal_ch2, sampling_rate=record.fs, show=False)

        # Smooth signals
        signal_smoothed_ch1 = ecg_ch1['filtered']
        signal_smoothed_ch2 = ecg_ch2['filtered']

        # Reading r-peaks
        r_peaks = ecg_ch1['rpeaks']

        # Reading annotations. `symbol` and `sample` are labels and values respectively.
        ann_symbol = annotation.symbol
        ann_sample = annotation.sample

        print(len(ann_sample), ann_symbol, r_peaks)

        # Iterate annotations
        for idx, symbol in enumerate(ann_symbol):
            if symbol in BEATS:
                ann_idx = ann_sample[idx]
                if ann_idx - left_range >= 0 and ann_idx + right_range < record.sig_len:
                    if symbol in N:
                        closest_r_peak = r_peaks[np.argmin(np.abs(r_peaks - ann_idx))]
                        if abs(closest_r_peak - ann_idx) < 10:
                            # samples.append(([signal1[ann_idx - left_range:ann_idx + right_range],
                            #                  signal2[ann_idx - left_range:ann_idx + right_range]], 'N', symbol))
                            samples.append([signal1[ann_idx - left_range:ann_idx + right_range],
                                            signal2[ann_idx - left_range:ann_idx + right_range]])
                            # labels.append(('N', symbol))
                            labels.append('N')
                    else:
                        aami_label = ''
                        if symbol in S:
                            aami_label = 'S'
                        elif symbol in V:
                            aami_label = 'V'
                        elif symbol in F:
                            aami_label = 'F'
                        elif symbol in Q:
                            aami_label = 'Q'
                        else:
                            raise ValueError('Invalid annotation type!')

                        # samples.append(([signal1[ann_idx - left_range:ann_idx + right_range], signal2[ann_idx - left_range:ann_idx + right_range]],
                        #                 aami_label, symbol))
                        samples.append([signal1[ann_idx - left_range:ann_idx + right_range],
                                        signal2[ann_idx - left_range:ann_idx + right_range]])
                        # labels.append((aami_label, symbol))
                        labels.append(aami_label)


def process_uci_har(data_path):
    trainX = pd.read_csv(os.path.join(data_path, 'train/X_train.txt'), delim_whitespace=True, header=None)
    trainy = pd.read_csv(os.path.join(data_path, 'train/y_train.txt'), delim_whitespace=True, header=None)
    train_subj = pd.read_csv(os.path.join(data_path, 'train/subject_train.txt'), delim_whitespace=True, header=None)
    testX = pd.read_csv(os.path.join(data_path, 'test/X_test.txt'), delim_whitespace=True, header=None)
    testy = pd.read_csv(os.path.join(data_path, 'test/y_test.txt'), delim_whitespace=True, header=None)
    test_subj = pd.read_csv(os.path.join(data_path, 'test/subject_test.txt'), delim_whitespace=True, header=None)


if __name__ == '__main__':
    print(f'[INFO] Start processing mit arrhythmia dataset...')
    # process_mit_arrhythmia(MIT_ARRHYTHMIA_PATH)

    print(f'[INFO] Start processing mit atrial dataset...')
    process_mit_atrial(MIT_ATRIAL_PATH)
