#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Xiao Qinfeng
# @Date:   2021/8/9 11:16
# @Last Modified by:   Xiao Qinfeng
# @Last Modified time: 2021/8/9 11:16
# @Software: PyCharm

import os
import warnings
from typing import List

import torch
import numpy as np
import scipy.io as sio
from tqdm.std import tqdm
from torch.utils.data import Dataset


class AMIGOSDataset(Dataset):

    num_subject = 40
    sampling_rate = 128

    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0, transform=None):
        self.transform = transform
        self.label_dim = label_dim

        files = sorted(os.listdir(data_path))
        assert len(files) == self.num_subject
        files = [files[i] for i in subject_list]

        all_data = []
        all_labels = []
        for a_file in tqdm(files):
            data = sio.loadmat(os.path.join(data_path, a_file))

            subject_data = []
            subject_label = []
            for i in range(data['joined_data'].shape[1]):
                trial_data = data['joined_data'][0, i]
                trial_label = data['labels_selfassessment'][0, i]
                trial_data = trial_data[:trial_data.shape[0] // self.sampling_rate * self.sampling_rate]
                trial_data = trial_data.reshape(trial_data.shape[0] // self.sampling_rate, self.sampling_rate,
                                                trial_data.shape[-1])
                trial_data = np.swapaxes(trial_data, 1, 2)

                if np.isnan(trial_data).any():
                    warnings.warn(
                        f"The array of {a_file} - {i} contains {np.sum(np.isnan(trial_data))} NaN of total {np.prod(trial_data.shape)} points, dropped.")
                    # trial_data[np.isnan(trial_data)] = 0
                    continue

                if trial_data.shape[0] % num_seq != 0:
                    trial_data = trial_data[:trial_data.shape[0] // num_seq * num_seq]

                # Standardize
                mean_value = np.expand_dims(trial_data.mean(axis=0), axis=0)
                std_value = np.expand_dims(trial_data.std(axis=0), axis=0)
                trial_data = (trial_data - mean_value) / std_value

                trial_data = trial_data.reshape(trial_data.shape[0] // num_seq, num_seq, *trial_data.shape[1:])

                if 0 in trial_data.shape:
                    warnings.warn(f"The array of shape {data['joined_data'][0, i].shape} is too small, dropped.")
                    continue

                trial_label = np.repeat(trial_label, trial_data.shape[1], axis=0)
                trial_label = np.repeat(np.expand_dims(trial_label, axis=0), trial_data.shape[0], axis=0)

                if 0 in trial_label.shape:
                    warnings.warn(f"The label of {a_file} - {i} is malfunctioned, dropped.")
                    continue

                subject_data.append(trial_data)
                subject_label.append(trial_label)

            subject_data = np.concatenate(subject_data, axis=0)
            subject_label = np.concatenate(subject_label, axis=0)

            all_data.append(subject_data)
            all_labels.append(subject_label)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        print(all_data.shape)
        print(all_labels.shape)

        self.data = all_data
        self.labels = all_labels

    def __getitem__(self, item):
        x = self.data[item].astype(np.float32)
        label = self.labels[item].astype(np.long)[:, self.label_dim]
        y = np.zeros_like(label, dtype=np.long)
        y[label >= 5] = 1

        if self.transform is not None:
            x = self.transform(x)

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data)
