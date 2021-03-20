"""
@Time    : 2021/2/13 12:58
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : preprocessing.py
@Software: PyCharm
@Desc    : 
"""
import os
import warnings
import argparse
from typing import List
from tqdm.std import tqdm

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm.std import tqdm

import torch
import torchvision


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-name', type=str, choices=['deap', 'amigos'], required=True)
    parser.add_argument('--data-path', type=str, required=True)

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


def process_deap(data_path: str):
    num_trial = 40
    sampling_rate = 128
    fps = 50

    if not os.path.exists(os.path.join(data_path, 'processed')):
        warnings.warn(f"The path {os.path.join(data_path, 'processed')} dose not exists, created...")
        os.makedirs(os.path.join(data_path, 'processed'))

    participants = sorted(os.listdir(os.path.join(data_path, 'video')))

    meta_dict = {'path': [], 'patient': [], 'trial': []}
    for patient in participants:
        print(f'[INFO] Processing patient {patient} ...')
        signal_dict = sio.loadmat(os.path.join(data_path, 'signal', f'{patient}.mat'))
        signal_data = signal_dict['data']
        label = signal_dict['labels']

        for trial in range(1, num_trial+1):
            dest_path = os.path.join(data_path, f'processed/{patient}/{trial}')

            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            print(f'[INFO] Processing trial {trial} ...')
            signal_trial = signal_data[trial-1]
            label_trial = label[trial-1]
            try:
                vframes, aframes, meta_info = torchvision.io.read_video(os.path.join(data_path, 'video', patient, f'{patient}_trial{trial:02d}.avi'), pts_unit='sec')
            except FileNotFoundError as e:
                print(e)
                print(f'[INFO] The video file of patient {patient}, trial {trial} is lost...')
                continue
            video_trial = vframes.numpy()

            signal_trial = signal_trial[:, :sampling_rate*60].reshape(signal_trial.shape[0], 60, sampling_rate)
            signal_trial = np.swapaxes(signal_trial, 0, 1)
            try:
                video_trial = video_trial.reshape(60, fps, *video_trial.shape[1:])
            except ValueError as e:
                print(e)
                print(f'[INFO] The video file of patient {patient}, trial {trial} is malfunctioned...')
                continue

            # print(signal_trial.shape, video_trial.shape)
            for i in range(60):
                meta_dict['path'].append(os.path.join(dest_path, f'{i}.npz'))
                meta_dict['patient'].append(patient)
                meta_dict['trial'].append(trial)
                np.savez(os.path.join(dest_path, f'{i}.npz'), signal=signal_trial[i], video=video_trial[i], label=label)

    meta_df = pd.DataFrame(meta_dict)
    meta_df.to_csv(os.path.join(data_path, 'processed/meta.csv'))


def process_amigos(data_path: str):
    pass


if __name__ == '__main__':
    args = parse_args()

    assert os.path.exists(os.path.join(args.data_path, 'video')), 'The video files must be stored in folder `video`!'
    assert os.path.exists(os.path.join(args.data_path, 'signal')), 'The signal files must be stored in folder `signal`!'

    if args.data_name == 'deap':
        process_deap(args.data_path)
    elif args.data_name == 'amigos':
        process_amigos(args.data_path)
    else:
        raise ValueError
