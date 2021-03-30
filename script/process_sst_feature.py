"""
@Time    : 2021/3/22 23:26
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : process_sst_feature.py
@Software: PyCharm
@Desc    : 
"""
import os
import itertools
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm.std import tqdm
from scipy import interpolate

SEED_NUM_SUBJECT = 45
SEED_SAMPLING_RATE = 200
SEED_LABELS = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]

SEED_IV_NUM_SUBJECT = 15
SEED_IV_SAMPLING_RATE = 200

DEAP_NUM_SUBJECT = 32
DEAP_SAMPLING_RATE = 128


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--raw-path', type=str, default='/data/DataHub/EmotionRecognition/SEED/Preprocessed_EEG')
    parser.add_argument('--feature-path', type=str, default='/data/DataHub/EmotionRecognition/SEED/ExtractedFeatures')
    parser.add_argument('--dest-path', type=str, required=True)
    parser.add_argument('--data-name', type=str, default='SEED', choices=['SEED', 'DEAP', 'SEED-IV'])

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


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(os.path.join(args.dest_path, 'raw')):
        os.makedirs(os.path.join(args.dest_path, 'raw'))
    if not os.path.exists(os.path.join(args.dest_path, 'feature')):
        os.makedirs(os.path.join(args.dest_path, 'feature'))

    if args.data_name == 'SEED-IV':
        for i in range(3):
            if not os.path.exists(os.path.join(args.dest_path, 'raw', f'{i + 1}')):
                os.makedirs(os.path.join(args.dest_path, 'raw', f'{i + 1}'))
            if not os.path.exists(os.path.join(args.dest_path, 'feature', f'{i + 1}')):
                os.makedirs(os.path.join(args.dest_path, 'feature', f'{i + 1}'))

    channel_map = {}

    num_session = 1

    if args.data_name == 'SEED' or args.data_name == 'SEED-IV':
        for i in range(3):
            channel_map[i] = (0, 3 + i)

        channel_map[3] = (1, 3)
        channel_map[4] = (1, 5)

        for i in range(5, 50):
            channel_map[i] = (2 + (i - 5) // 9, (i - 5) % 9)

        for i in range(50, 57):
            channel_map[i] = (7, i - 50 + 1)

        for i in range(57, 62):
            channel_map[i] = (8, i - 57 + 2)

        if args.data_name == 'SEED-IV':
            num_session = 3
    else:
        raise ValueError

    grid_mat = np.zeros(shape=(9, 9))
    for i in range(62):
        grid_mat[channel_map[i][0], channel_map[i][1]] = 1
    # plt.imshow(grid_mat)
    # plt.show()

    all_raw_data = []
    all_feature_data = []

    # Enumerate all files
    for i_session in range(1, num_session + 1):
        print(f'[INFO] Processing {i_session}-th session...')
        if num_session == 1:
            files = sorted(os.listdir(args.raw_path))
            assert len(files) == SEED_NUM_SUBJECT
        else:
            files = sorted(os.listdir(os.path.join(args.raw_path, f'{i_session}')))
            assert len(files) == SEED_IV_NUM_SUBJECT

        for a_file in files:
            print(f'[INFO] Processing file {a_file}...')

            if num_session == 1:
                assert os.path.exists(os.path.join(args.feature_path, a_file))
                raw_dict = sio.loadmat(os.path.join(args.raw_path, a_file))
                feature_dict = sio.loadmat(os.path.join(args.feature_path, a_file))
            else:
                assert os.path.exists(os.path.join(args.feature_path, f'{i_session}', a_file))
                raw_dict = sio.loadmat(os.path.join(args.raw_path, f'{i_session}', a_file))
                feature_dict = sio.loadmat(os.path.join(args.feature_path, f'{i_session}', a_file))

            print(raw_dict.keys())
            print(feature_dict.keys())

            subject_raw_data = {}

            print(f'[INFO] Processing raw data...')
            for key in raw_dict.keys():
                if 'eeg' in key:
                    current_data = raw_dict[key]
                    current_data = current_data[:, :-1]  # Exclude the last point

                    current_data = (current_data - current_data.mean(axis=0)) / current_data.std(axis=0)

                    points = np.array([value[0] * 9 + value[1] for value in channel_map.values()])
                    values = np.zeros((9, 9))

                    trial_raw_data = []

                    for ts in tqdm(range(current_data.shape[-1]), desc=key):
                        np.put(values, points, current_data[:, ts])

                        ############### Only for test ###############
                        # plt.imshow(values)
                        # plt.title('original_values')
                        # plt.colorbar()
                        # plt.show()
                        ############### Only for test ###############

                        #             grid_x, grid_y = np.meshgrid(np.arange(9), np.arange(9))
                        #             interpolated_mat = interpolate.griddata(points, values, (grid_x, grid_y), method='cubic')
                        #             interpolator = interpolate.interp2d(x=points[:,1], y=points[:,0], z=values, kind='cubic')
                        interpolator = interpolate.interp2d(x=np.arange(9), y=np.arange(9), z=values, kind='cubic')
                        interpolated_mat = interpolator(np.linspace(0, 8, 32), np.linspace(0, 8, 32))

                        ############### Only for test ###############
                        # print(points.shape)
                        # print(interpolated_mat)
                        # plt.imshow(interpolated_mat)
                        # plt.title('interpolated_mat')
                        # plt.colorbar()
                        # plt.show()
                        ############### Only for test ###############

                        trial_raw_data.append(interpolated_mat)
                    trial_raw_data = np.stack(trial_raw_data, axis=0)
                    subject_raw_data[key] = trial_raw_data
            if num_session == 1:
                # sio.savemat(os.path.join(args.dest_path, 'raw', a_file), subject_raw_data)
                np.savez(os.path.join(args.dest_path, 'raw', a_file), **subject_raw_data)
            else:
                # sio.savemat(os.path.join(args.dest_path, 'raw', f'{i_session}', a_file), subject_raw_data)
                np.savez(os.path.join(args.dest_path, 'raw', f'{i_session}', a_file), **subject_raw_data)

            subject_feature_data = {}

            print(f'[INFO] Processing feature data...')
            for key in feature_dict.keys():
                if key.startswith('de_LDS'):
                    current_data = feature_dict[key]
                    current_data = (current_data - current_data.mean(axis=1)[:, np.newaxis, :]) / current_data.std(
                        axis=1)[
                                                                                                  :, np.newaxis, :]
                    points = np.array([value[0] * 9 + value[1] for value in channel_map.values()])
                    values = np.zeros((9, 9))

                    trial_feature_data = []

                    for ts in tqdm(range(current_data.shape[1]), desc=key):
                        band_feature_data = []
                        for i_feature in range(current_data.shape[-1]):
                            np.put(values, points, current_data[:, ts, i_feature])

                            ############### Only for test ###############
                            # plt.imshow(values)
                            # plt.title('original_values')
                            # plt.colorbar()
                            # plt.show()
                            ############### Only for test ###############

                            interpolator = interpolate.interp2d(x=np.arange(9), y=np.arange(9), z=values, kind='cubic')
                            interpolated_mat = interpolator(np.linspace(0, 8, 32), np.linspace(0, 8, 32))

                            ############### Only for test ###############
                            # print(points.shape)
                            # print(interpolated_mat)
                            # plt.imshow(interpolated_mat)
                            # plt.title('interpolated_mat')
                            # plt.colorbar()
                            # plt.show()
                            ############### Only for test ###############

                            band_feature_data.append(interpolated_mat)
                        band_feature_data = np.stack(band_feature_data, axis=-1)
                        # print(band_feature_data.shape)
                        trial_feature_data.append(band_feature_data)
                    trial_feature_data = np.stack(trial_feature_data, axis=-2)
                    # print(trial_feature_data.shape)
                    subject_feature_data[key] = trial_feature_data
            if num_session == 1:
                # sio.savemat(os.path.join(args.dest_path, 'feature', a_file), subject_feature_data)
                np.savez(os.path.join(args.dest_path, 'feature', a_file), **subject_feature_data)
            else:
                # sio.savemat(os.path.join(args.dest_path, 'feature', f'{i_session}', a_file), subject_feature_data)
                np.savez(os.path.join(args.dest_path, 'feature', f'{i_session}', a_file), **subject_feature_data)
