"""
@Time    : 2021/2/6 15:20
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import os
import warnings
from typing import List

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from torch.utils.data import Dataset
from tqdm.std import tqdm

# from .utils import tensor_standardize
from .seed import SEEDDataset
from .seediv import SEEDIVDataset
from .deap import DEAPDataset
from .amigos import AMIGOSDataset








ISRUC_NUM_SUBJECT = 99

SLEEPEDF_NUM_SUBJECT = 153

EPS = 1e-8


def tackle_denominator(x: np.ndarray):
    x[x == 0.0] = EPS
    return x


def tensor_standardize(x: np.ndarray, dim=-1):
    x_mean = np.expand_dims(x.mean(axis=dim), axis=dim)
    x_std = np.expand_dims(x.std(axis=dim), axis=dim)
    return (x - x_mean) / tackle_denominator(x_std)











# class SEEDSSTDataset(Dataset):
#     def __init__(self, data_path, num_seq, subject_list: List, label_dim=0):
#         self.data_path = data_path
#         self.num_seq = num_seq
#         self.label_dim = label_dim
#
#         subjects = sorted(os.listdir(data_path))
#         assert len(subjects) == SEED_NUM_SUBJECT
#         subjects = [subjects[i] for i in subject_list]
#
#         all_files = []
#
#         for a_subject in subjects:
#             files = sorted(os.listdir(os.path.join(data_path, a_subject)))
#             all_files += list(zip([a_subject] * len(files), files))
#
#         self.all_files = all_files
#
#     def __getitem__(self, item):
#         subject_name, filename = self.all_files[item]
#         data = np.load(os.path.join(self.data_path, subject_name, filename))
#         x = data['data'].astype(np.float32)
#         y = data['label'].astype(np.long)
#
#         return torch.from_numpy(x), torch.from_numpy(y)
#
#     def __len__(self):
#         return len(self.all_files)
#
#
# class SEEDIVSSTDataset(Dataset):
#     def __init__(self, data_path, num_seq, subject_list: List, label_dim=0):
#         self.data_path = data_path
#         self.num_seq = num_seq
#         self.label_dim = label_dim
#
#         subjects = sorted(os.listdir(os.path.join(data_path, '1'))) + \
#                    sorted(os.listdir(os.path.join(data_path, '2'))) + \
#                    sorted(os.listdir(os.path.join(data_path, '3')))
#         subjects_with_session = list(zip([1] * 15 + [2] * 15 + [3] * 15, subjects))
#         assert len(subjects_with_session) == SEED_IV_NUM_SUBJECT
#         subjects_with_session = [subjects_with_session[i] for i in subject_list]
#
#         all_files = []
#
#         for a_session, a_subject in subjects_with_session:
#             files = sorted(os.listdir(os.path.join(data_path, f'{a_session}', a_subject)))
#             all_files += list(zip([a_session] * len(files), [a_subject] * len(files), files))
#
#         self.all_files = all_files
#
#     def __getitem__(self, item):
#         session, subject_name, filename = self.all_files[item]
#         data = np.load(os.path.join(self.data_path, f'{session}', subject_name, filename))
#         x = data['data'].astype(np.float32)
#         y = data['label'].astype(np.long)
#
#         return torch.from_numpy(x), torch.from_numpy(y)
#
#     def __len__(self):
#         return len(self.all_files)


class SleepDatasetImg(Dataset):
    def __init__(self, data_path, data_name, num_epoch, transform, patients: List = None, return_idx=False,
                 verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.num_epoch = num_epoch
        self.transform = transform
        self.patients = patients
        self.return_idx = return_idx

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            if verbose:
                print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            recordings = data['data']
            annotations = data['label']

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

        # x = torch.stack([self.transform(x[0]), self.transform(x[1])], dim=0)
        # print(x.shape, x[:, 0].shape, '-------------')
        if self.data_name == 'isruc':
            x_all = []
            for k in range(x.shape[1]):
                x_all.append(torch.stack([self.transform(Image.fromarray(x[i][k])) for i in range(x.shape[0])], dim=0))
            x = torch.cat(x_all, dim=1)
        else:
            x1 = torch.stack([self.transform(Image.fromarray(x[i][0])) for i in range(x.shape[0])],
                             dim=0)  # TODO for temp
            x2 = torch.stack([self.transform(Image.fromarray(x[i][1])) for i in range(x.shape[0])], dim=0)
            x = torch.cat([x1, x2], dim=1)
        y = torch.from_numpy(y.astype(np.long))

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
    # Instance: {}
    Shape of an Instance: {}
    Selected patients: {}
    **********************************************************************
                """.format(len(self.data), self.full_shape, self.patients)


class TwoDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)

        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, item):
        return (*self.dataset1[item], *self.dataset2[item])

    def __len__(self):
        return len(self.dataset1)


if __name__ == '__main__':
    base_path = 'data/sst_feature/SEED-IV/feature'

    # SST features
    # data = SEEDSSTDataset(os.path.join(base_path), num_seq=10,
    #                          subject_list=[i for i in range(SEED_NUM_SUBJECT // 10 * 9)])
    # data = SEEDIVSSTDataset(os.path.join(base_path), num_seq=10,
    #                            subject_list=[i for i in range(SEED_NUM_SUBJECT // 10 * 9)])

    # base_path = '/data/DataHub/EmotionRecognition'

    # Raw features
    # data = DEAPDataset(os.path.join(base_path, 'DEAP', 'signal'), num_seq=0,
    #                       subject_list=[i for i in range(DEAP_NUM_SUBJECT // 10 * 9)])
    # data = SEEDDataset(os.path.join(base_path, 'SEED', 'Preprocessed_EEG'), num_seq=0,
    #                       subject_list=[i for i in range(SEED_NUM_SUBJECT // 10 * 9)])
    # data = SEEDIVDataset(os.path.join(base_path, 'SEED-IV', 'eeg_raw_data'), num_seq=0,
    #                         subject_list=[i for i in range(SEED_IV_NUM_SUBJECT // 10 * 9)])
    # data = AMIGOSDataset(os.path.join(base_path, 'AMIGOS', 'signal'), num_seq=10,
    #                         subject_list=[i for i in range(AMIGOS_NUM_SUBJECT // 10 * 9)])
    # print(dataset[np.random.randint(low=0, high=len(dataset))][0].shape,
    #       dataset[np.random.randint(low=0, high=len(dataset))][1].shape)
