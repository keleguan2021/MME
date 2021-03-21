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
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from tqdm.std import tqdm

# from .utils import tensor_standardize


SEED_NUM_SUBJECT = 45
SEED_SAMPLING_RATE = 200
SEED_LABELS = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]

DEAP_NUM_SUBJECT = 32
DEAP_SAMPLING_RATE = 128


class SEEDDataset(Dataset):
    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0):
        files = sorted(os.listdir(data_path))
        assert len(files) == SEED_NUM_SUBJECT
        files = [files[i] for i in subject_list]

        all_data = []
        all_label = []
        # Enumerate all files
        for a_file in tqdm(files):
            data = sio.loadmat(os.path.join(data_path, a_file))
            # Each file contains 15 consecutive trials
            movie_ids = sorted(list(filter(lambda x: not x.startswith('__'), data.keys())))
            subject_data = []
            subject_label = []
            assert len(movie_ids) == len(SEED_LABELS)

            for i, key in enumerate(movie_ids):
                trial_data = data[key]
                trial_data = trial_data[:, :-1]  # remove the last redundant point
                # trial_data = tensor_standardize(trial_data, dim=-1)
                assert trial_data.shape[1] % SEED_SAMPLING_RATE == 0

                trial_data = trial_data.reshape(trial_data.shape[0], trial_data.shape[1] // SEED_SAMPLING_RATE,
                                                SEED_SAMPLING_RATE)
                trial_data = np.swapaxes(trial_data, 0, 1)
                # Shape: (num_seq, channel, time_len)

                if trial_data.shape[0] % num_seq != 0:
                    trial_data = trial_data[:trial_data.shape[0] // num_seq * num_seq]
                trial_data = trial_data.reshape(trial_data.shape[0] // num_seq, num_seq, *trial_data.shape[1:])
                trial_label = np.full(shape=trial_data.shape[:2], fill_value=SEED_LABELS[i])

                # Final shape: (num_sample, num_seq, channel, time_len)
                subject_data.append(trial_data)
                subject_label.append(trial_label)
            subject_data = np.concatenate(subject_data, axis=0)
            subject_label = np.concatenate(subject_label, axis=0)
            all_data.append(subject_data)
            all_label.append(subject_label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        print(all_data.shape)
        print(all_label.shape)

        self.data = all_data
        self.labels = all_label

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        y = self.labels[idx].astype(np.long)

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"""
        Dataset SEED:
        # of samples - {len(self.data)}
        """


class DEAPDataset(Dataset):
    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0):
        self.label_dim = label_dim

        files = sorted(os.listdir(data_path))
        assert len(files) == DEAP_NUM_SUBJECT
        files = [files[i] for i in subject_list]

        all_data = []
        all_labels = []
        for a_file in tqdm(files):
            data = sio.loadmat(os.path.join(data_path, a_file))
            subject_data = data['data']  # trial x channel x data
            subject_label = data['labels']  # trial x label (valence, arousal, dominance, liking)
            # subject_data = tensor_standardize(subject_data, dim=-1)

            subject_data = subject_data.reshape(*subject_data.shape[:2], subject_data.shape[-1] // DEAP_SAMPLING_RATE,
                                                DEAP_SAMPLING_RATE)
            subject_data = np.swapaxes(subject_data, 1, 2)

            if subject_data.shape[1] % num_seq != 0:
                subject_data = subject_data[:, :subject_data.shape[1] // num_seq * num_seq]
            subject_data = subject_data.reshape(subject_data.shape[0], subject_data.shape[1] // num_seq, num_seq,
                                                *subject_data.shape[-2:])

            subject_label = np.repeat(np.expand_dims(subject_label, axis=1), subject_data.shape[1], axis=1)
            subject_label = np.repeat(np.expand_dims(subject_label, axis=2), subject_data.shape[2], axis=2)

            subject_data = subject_data.reshape(subject_data.shape[0] * subject_data.shape[1], *subject_data.shape[2:])
            subject_label = subject_label.reshape(subject_label.shape[0] * subject_label.shape[1],
                                                  *subject_label.shape[2:])

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
        y = self.labels[item].astype(np.long)[:, self.label_dim]

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data)


class AMIGOSDataset(Dataset):
    def __init__(self, data_path, num_seq, subject_list: List):
        files = sorted(os.listdir(data_path))


if __name__ == '__main__':
    base_path = '/data/DataHub/EmotionRecognition'

    dataset = DEAPDataset(os.path.join(base_path, 'DEAP', 'signal'), num_seq=10,
                          subject_list=[i for i in range(DEAP_NUM_SUBJECT // 10 * 9)])
    # dataset = SEEDDataset(os.path.join(base_path, 'SEED', 'Preprocessed_EEG'), num_seq=10,
    #                       subject_list=[i for i in range(SEED_NUM_SUBJECT // 10 * 9)])
    print(dataset)
