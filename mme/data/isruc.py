#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Xiao Qinfeng
# @Date:   2021/8/9 11:16
# @Last Modified by:   Xiao Qinfeng
# @Last Modified time: 2021/8/9 11:16
# @Software: PyCharm

import os
from typing import List

import numpy as np
import scipy.io as sio
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import Dataset

from .utils import tensor_standardize


class ISRUCDataset(Dataset):
    num_subject = 100
    fs = 200

    def __init__(self, data_path, num_epoch, transform=None, patients: List = None, preprocessing: str = 'none',
                 modal='eeg',
                 return_idx=False, verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.transform = transform
        self.patients = patients
        self.preprocessing = preprocessing
        self.modal = modal
        self.return_idx = return_idx

        assert preprocessing in ['none', 'quantile', 'standard']
        assert modal in ['eeg', 'emg', 'eog']

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            if verbose:
                print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = sio.loadmat(os.path.join(data_path, patient))
            if modal == 'eeg':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
            elif modal == 'emg':
                recordings = np.stack([data['X1'], data['X3']], axis=1)
            elif modal == 'eog':
                recordings = np.stack([data['LOC_A2'], data['ROC_A1']], axis=1)
            else:
                raise ValueError

            annotations = data['label'].flatten()

            if preprocessing == 'standard':
                # print(f'[INFO] Applying standard scaler...')
                # scaler = StandardScaler()
                # recordings_old = recordings
                # recordings = []
                # for j in range(recordings_old.shape[0]):
                #     recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                # recordings = np.stack(recordings, axis=0)

                recordings = tensor_standardize(recordings, dim=-1)
            elif preprocessing == 'quantile':
                # print(f'[INFO] Applying quantile scaler...')
                scaler = QuantileTransformer(output_distribution='normal')
                recordings_old = recordings
                recordings = []
                for j in range(recordings_old.shape[0]):
                    recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                recordings = np.stack(recordings, axis=0)
            else:
                pass

            if verbose:
                print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = x.astype(np.float32)
        y = y.astype(np.long)

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return """
**********************************************************************
Dataset Summary:
Preprocessing: {}
# Instance: {}
Shape of an Instance: {}
Selected patients: {}
**********************************************************************
            """.format(self.preprocessing, len(self.data), self.full_shape, self.patients)

    @property
    def channels(self):
        return self.data.shape[2]
