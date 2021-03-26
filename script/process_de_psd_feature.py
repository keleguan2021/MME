"""
@Time    : 2021/3/25 10:33
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : process_de_psd_feature.py
@Software: PyCharm
@Desc    : 
"""
import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy.fftpack import fft, ifft
from tqdm.std import tqdm


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--raw-path', type=str, default='/data/DataHub/EmotionRecognition/DEAP/signal')
    parser.add_argument('--dest-path', type=str, required=True)
    parser.add_argument('--data-name', type=str, default='DEAP', choices=['SEED', 'DEAP', 'SEED-IV'])

    args_parsed = parser.parse_args()

    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)

    return args_parsed


def DE_PSD(data, sampling_rate, freq_start, freq_end, fs, window_size):
    # data (channel, time_len)
    WindowPoints = fs * window_size

    fStartNum = np.zeros([len(freq_start)], dtype=int)
    fEndNum = np.zeros([len(freq_end)], dtype=int)
    for i in range(0, len(freq_start)):
        fStartNum[i] = int(freq_start[i] / fs * sampling_rate)
        fEndNum[i] = int(freq_end[i] / fs * sampling_rate)

    n = data.shape[0]
    m = data.shape[1]

    # print(m,n,l)
    psd = np.zeros([n, len(freq_start)])
    de = np.zeros([n, len(freq_start)])
    # Hanning window
    Hlength = window_size * fs
    # Hwindow=hanning(Hlength)
    Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])

    WindowPoints = fs * window_size
    dataNow = data[0:n]
    for j in range(0, n):
        temp = dataNow[j]
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, sampling_rate)
        magFFTdata = abs(FFTdata[0:int(sampling_rate / 2)])
        for p in range(0, len(freq_start)):
            E = 0
            # E_log = 0
            for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                E = E + magFFTdata[p0] * magFFTdata[p0]
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            psd[j][p] = E
            de[j][p] = math.log(100 * E, 2)
            # de(j,i,p)=log2((1+E)^4)

    return psd, de


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)
