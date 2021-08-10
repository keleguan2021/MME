#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Xiao Qinfeng
# @Date:   2021/8/9 11:15
# @Last Modified by:   Xiao Qinfeng
# @Last Modified time: 2021/8/9 11:15
# @Software: PyCharm

import os
from typing import List

import torch
import numpy as np
import scipy.io as sio
from tqdm.std import tqdm
from torch.utils.data import Dataset


class SEEDDataset(Dataset):
    num_subject = 45
    fs = 200
    raw_labels = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]

    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0, transform=None):
        self.transform = transform

        files = sorted(os.listdir(data_path))
        assert len(files) == self.num_subject
        files = [files[i] for i in subject_list]

        all_data = []
        all_label = []
        # Enumerate all files
        for a_file in tqdm(files):
            data = sio.loadmat(os.path.join(data_path, a_file))
            # Each file contains 15 consecutive trials
            movie_ids = list(filter(lambda x: not x.startswith('__'), data.keys()))
            subject_data = []
            subject_label = []
            assert len(movie_ids) == len(self.raw_labels)

            for i, key in enumerate(movie_ids):
                trial_data = data[key]
                trial_data = trial_data[:, :-1]  # remove the last redundant point
                # trial_data = tensor_standardize(trial_data, dim=-1)
                assert trial_data.shape[1] % self.sampling_rate == 0

                trial_data = trial_data.reshape(trial_data.shape[0], trial_data.shape[1] // self.sampling_rate,
                                                self.sampling_rate)
                trial_data = np.swapaxes(trial_data, 0, 1)
                # Shape: (num_seq, channel, time_len)

                if num_seq == 0:
                    trial_data = np.expand_dims(trial_data, axis=1)
                else:
                    if trial_data.shape[0] % num_seq != 0:
                        trial_data = trial_data[:trial_data.shape[0] // num_seq * num_seq]
                    trial_data = trial_data.reshape(trial_data.shape[0] // num_seq, num_seq, *trial_data.shape[1:])

                trial_label = np.full(shape=trial_data.shape[:2], fill_value=self.raw_labels[i])

                # Final shape: (num_sample, num_seq, channel, time_len)
                subject_data.append(trial_data)
                subject_label.append(trial_label)
            subject_data = np.concatenate(subject_data, axis=0)
            subject_label = np.concatenate(subject_label, axis=0)
            all_data.append(subject_data)
            all_label.append(subject_label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)

        if num_seq == 0:
            all_data = np.squeeze(all_data)
            # all_label = np.squeeze(all_label)

        print(all_data.shape)
        print(all_label.shape)

        self.data = all_data
        self.labels = all_label

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        y = self.labels[idx].astype(np.long)

        if self.transform is not None:
            x = self.transform(x)

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"""
        Dataset SEED:
        # of samples - {len(self.data)}
        """
