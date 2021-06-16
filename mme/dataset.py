"""
@Time    : 2021/2/6 15:20
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dataset.py
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
from sklearn.preprocessing import QuantileTransformer

# from .utils import tensor_standardize


SEED_NUM_SUBJECT = 45
SEED_SAMPLING_RATE = 200
SEED_LABELS = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]

SEED_IV_NUM_SUBJECT = 45
SEED_IV_SAMPLING_RATE = 200
SEED_IV_LABELS = [
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
]

DEAP_NUM_SUBJECT = 32
DEAP_SAMPLING_RATE = 128

AMIGOS_NUM_SUBJECT = 40
AMIGOS_SAMPLING_RATE = 128

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
            movie_ids = list(filter(lambda x: not x.startswith('__'), data.keys()))
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

                if num_seq == 0:
                    trial_data = np.expand_dims(trial_data, axis=1)
                else:
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

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"""
        Dataset SEED:
        # of samples - {len(self.data)}
        """


class SEEDIVDataset(Dataset):
    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0):
        files = sorted(os.listdir(os.path.join(data_path, '1'))) + \
                sorted(os.listdir(os.path.join(data_path, '2'))) + \
                sorted(os.listdir(os.path.join(data_path, '3')))
        files_with_session = list(zip([1] * 15 + [2] * 15 + [3] * 15, files))
        assert len(files_with_session) == SEED_IV_NUM_SUBJECT
        files_with_session = [files_with_session[i] for i in subject_list]
        print(files_with_session)

        all_data = []
        all_label = []
        # Enumerate all files
        for i_session, a_file in tqdm(files_with_session):
            data = sio.loadmat(os.path.join(data_path, f'{i_session}', a_file))
            # Each file contains 15 consecutive trials
            movie_ids = list(filter(lambda x: not x.startswith('__'), data.keys()))
            subject_data = []
            subject_label = []
            assert len(movie_ids) == len(SEED_IV_LABELS[i_session - 1])

            for i, key in enumerate(movie_ids):
                trial_data = data[key]
                trial_data = trial_data[:, :-1]  # remove the last redundant point
                # trial_data = tensor_standardize(trial_data, dim=-1)
                assert trial_data.shape[1] % SEED_IV_SAMPLING_RATE == 0

                trial_data = trial_data.reshape(trial_data.shape[0], trial_data.shape[1] // SEED_IV_SAMPLING_RATE,
                                                SEED_IV_SAMPLING_RATE)
                trial_data = np.swapaxes(trial_data, 0, 1)
                # Shape: (num_seq, channel, time_len)

                if num_seq == 0:
                    trial_data = np.expand_dims(trial_data, axis=1)
                else:
                    if trial_data.shape[0] % num_seq != 0:
                        trial_data = trial_data[:trial_data.shape[0] // num_seq * num_seq]
                    trial_data = trial_data.reshape(trial_data.shape[0] // num_seq, num_seq, *trial_data.shape[1:])
                trial_label = np.full(shape=trial_data.shape[:2], fill_value=SEED_IV_LABELS[i_session - 1][i])

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

        self.data = all_data.astype(np.float32)
        self.labels = all_label.astype(np.long)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data)


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
                                                DEAP_SAMPLING_RATE)  # (trial, channel, num_sec, time_len)
            subject_data = np.swapaxes(subject_data, 1, 2)  # (trial, num_sec, channel, time_len)

            if num_seq == 0:
                subject_data = np.expand_dims(subject_data, axis=2)
            else:
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

        if num_seq == 0:
            all_data = np.squeeze(all_data)
            # all_labels = np.squeeze(all_labels)

        print(all_data.shape)
        print(all_labels.shape)

        self.data = all_data
        self.labels = all_labels

    def __getitem__(self, item):
        x = self.data[item].astype(np.float32)
        label = self.labels[item].astype(np.long)[:, self.label_dim]
        y = np.zeros_like(label, dtype=np.long)
        y[label >= 5] = 1

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data)


class AMIGOSDataset(Dataset):
    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0):
        self.label_dim = label_dim

        files = sorted(os.listdir(data_path))
        assert len(files) == AMIGOS_NUM_SUBJECT
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
                trial_data = trial_data[:trial_data.shape[0] // AMIGOS_SAMPLING_RATE * AMIGOS_SAMPLING_RATE]
                trial_data = trial_data.reshape(trial_data.shape[0] // AMIGOS_SAMPLING_RATE, AMIGOS_SAMPLING_RATE,
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

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data)


class SEEDSSTDataset(Dataset):
    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0):
        self.data_path = data_path
        self.num_seq = num_seq
        self.label_dim = label_dim

        subjects = sorted(os.listdir(data_path))
        assert len(subjects) == SEED_NUM_SUBJECT
        subjects = [subjects[i] for i in subject_list]

        all_files = []

        for a_subject in subjects:
            files = sorted(os.listdir(os.path.join(data_path, a_subject)))
            all_files += list(zip([a_subject] * len(files), files))

        self.all_files = all_files

    def __getitem__(self, item):
        subject_name, filename = self.all_files[item]
        data = np.load(os.path.join(self.data_path, subject_name, filename))
        x = data['data'].astype(np.float32)
        y = data['label'].astype(np.long)

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.all_files)


class SEEDIVSSTDataset(Dataset):
    def __init__(self, data_path, num_seq, subject_list: List, label_dim=0):
        self.data_path = data_path
        self.num_seq = num_seq
        self.label_dim = label_dim

        subjects = sorted(os.listdir(os.path.join(data_path, '1'))) + \
                   sorted(os.listdir(os.path.join(data_path, '2'))) + \
                   sorted(os.listdir(os.path.join(data_path, '3')))
        subjects_with_session = list(zip([1] * 15 + [2] * 15 + [3] * 15, subjects))
        assert len(subjects_with_session) == SEED_IV_NUM_SUBJECT
        subjects_with_session = [subjects_with_session[i] for i in subject_list]

        all_files = []

        for a_session, a_subject in subjects_with_session:
            files = sorted(os.listdir(os.path.join(data_path, f'{a_session}', a_subject)))
            all_files += list(zip([a_session] * len(files), [a_subject] * len(files), files))

        self.all_files = all_files

    def __getitem__(self, item):
        session, subject_name, filename = self.all_files[item]
        data = np.load(os.path.join(self.data_path, f'{session}', subject_name, filename))
        x = data['data'].astype(np.float32)
        y = data['label'].astype(np.long)

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.all_files)


class SleepDataset(Dataset):
    def __init__(self, data_path, data_name, num_epoch, patients: List = None, preprocessing: str = 'none', modal='eeg',
                 return_idx=False, verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
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
            data = np.load(os.path.join(data_path, patient))
            if data_name == 'sleepedf':
                if modal == 'eeg':
                    recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)
                elif modal == 'emg':
                    recordings = np.expand_dims(data['emg'], axis=1)
                elif modal == 'eog':
                    recordings = np.expand_dims(data['eog'], axis=1)
                else:
                    raise ValueError

                annotations = data['annotation']
            elif data_name == 'isruc':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
                annotations = data['label'].flatten()
            else:
                raise ValueError

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
                # print(f'[INFO] Convert the unit from V to uV...')
                recordings *= 1e6

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

        x = torch.from_numpy(x.astype(np.float32))
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
Preprocessing: {}
# Instance: {}
Shape of an Instance: {}
Selected patients: {}
**********************************************************************
            """.format(self.preprocessing, len(self.data), self.full_shape, self.patients)


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
    # dataset = SEEDSSTDataset(os.path.join(base_path), num_seq=10,
    #                          subject_list=[i for i in range(SEED_NUM_SUBJECT // 10 * 9)])
    dataset = SEEDIVSSTDataset(os.path.join(base_path), num_seq=10,
                               subject_list=[i for i in range(SEED_NUM_SUBJECT // 10 * 9)])

    # base_path = '/data/DataHub/EmotionRecognition'

    # Raw features
    # dataset = DEAPDataset(os.path.join(base_path, 'DEAP', 'signal'), num_seq=0,
    #                       subject_list=[i for i in range(DEAP_NUM_SUBJECT // 10 * 9)])
    # dataset = SEEDDataset(os.path.join(base_path, 'SEED', 'Preprocessed_EEG'), num_seq=0,
    #                       subject_list=[i for i in range(SEED_NUM_SUBJECT // 10 * 9)])
    # dataset = SEEDIVDataset(os.path.join(base_path, 'SEED-IV', 'eeg_raw_data'), num_seq=0,
    #                         subject_list=[i for i in range(SEED_IV_NUM_SUBJECT // 10 * 9)])
    # dataset = AMIGOSDataset(os.path.join(base_path, 'AMIGOS', 'signal'), num_seq=10,
    #                         subject_list=[i for i in range(AMIGOS_NUM_SUBJECT // 10 * 9)])
    print(dataset[np.random.randint(low=0, high=len(dataset))][0].shape,
          dataset[np.random.randint(low=0, high=len(dataset))][1].shape)
