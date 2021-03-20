"""
@Time    : 2021/2/6 15:20
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dataset.py
@Software: PyCharm
@Desc    : 
"""
import os
from typing import List

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm.std import tqdm

import torch

from torch.utils.data import Dataset, DataLoader


SEED_SAMPLING_RATE = 200


class SEEDDataset(Dataset):
    def __init__(self, data_path):
        files = sorted(os.listdir(data_path))
        all_data = []
        # Enumerate all files
        for a_file in tqdm(files):
            data = sio.loadmat(os.path.join(data_path, a_file))
            # Each file contains 15 consecutive trials
            movie_ids = sorted(list(filter(lambda x: not x.startswith('__'), data.keys())))
            subject_data = []
            for key in movie_ids:
                trial_data = data[key]
                trial_data = trial_data[:, :-1]  # remove the last redundant point
                assert trial_data.shape[1] % SEED_SAMPLING_RATE == 0

                trial_data = trial_data.reshape(trial_data.shape[0], trial_data.shape[1] // SEED_SAMPLING_RATE,
                                                SEED_SAMPLING_RATE)
                trial_data = np.swapaxes(trial_data, 0, 1)
                # Final shape: (num_seq, channel, time_len)
                subject_data.append(trial_data)
            subject_data = np.concatenate(subject_data, axis=0)
            all_data.append(subject_data)
        all_data = np.stack(all_data, axis=0)
        print(all_data.shape)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    base_path = '/data/DataHub/EmotionRecognition'

    dataset = SEEDDataset(os.path.join(base_path, 'SEED', 'Preprocessed_EEG'))
    print(dataset)
