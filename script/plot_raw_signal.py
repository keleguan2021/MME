"""
@Time    : 2021/4/8 21:40
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : plot_raw_signal.py
@Software: PyCharm
@Desc    : 
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from tqdm.std import tqdm

data = sio.loadmat('/data/DataHub/EmotionRecognition/SEED/Preprocessed_EEG/1_20131027.mat')
signal = data['djc_eeg1']
signal = signal[:, :signal.shape[1] // 2]
# ts = [f'{i*0.005}ms' for i in range(signal.shape[1])]
ts = np.arange(signal.shape[1])

with plt.style.context(['science']):
    plt.figure(figsize=(8, 6))
    for i in tqdm(range(signal.shape[0])):
        x = signal[i]
        x = (x - x.min()) / x.max()
        x = x + i + 0.2
        plt.plot(ts, x, linewidth=0.8)
        plt.xlim(xmin=0, xmax=signal.shape[1])
    plt.savefig('signal_raw.svg')
    plt.show()

data = sio.loadmat('/data/DataHub/EmotionRecognition/SEED/ExtractedFeatures/1_20131027.mat')
signal = data['de_LDS1']
print(signal.shape)

ts = [r'$\gamma$', r'$\beta$', r'$\alpha$', r'$\theta$', r'$\delta$']

with plt.style.context(['science']):
    plt.figure(figsize=(8, 6))
    dis = 0
    for i in tqdm(range(signal.shape[0])):
        x = signal[i, 0, :]
        # x = (x - x.min())/x.max()
        x = x + i + 2
        dis = x.max() - x.min()
        plt.plot(ts, x, linewidth=0.8)
        plt.xlim(xmin=0)
    plt.savefig('signal_de.svg')
    plt.show()
