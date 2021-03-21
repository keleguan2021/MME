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

SEED_NUM_SUBJECT = 45
SEED_SAMPLING_RATE = 200
SEED_LABELS = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]


class SEEDDataset(Dataset):
    def __init__(self, data_path, num_seq, subject_list: List):
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


if __name__ == '__main__':
    base_path = '/data/DataHub/EmotionRecognition'

    dataset = SEEDDataset(os.path.join(base_path, 'SEED', 'Preprocessed_EEG'), num_seq=10,
                          subject_list=[i for i in range(SEED_NUM_SUBJECT // 10 * 9)])
    print(dataset)
