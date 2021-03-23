"""
@Time    : 2021/3/22 23:26
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : process_sst_feature.py
@Software: PyCharm
@Desc    : 
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm.std import tqdm
from scipy import interpolate

SEED_NUM_SUBJECT = 45
SEED_SAMPLING_RATE = 200
SEED_LABELS = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
SEED_CHANNEL_MAP = {}

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

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)

    files = sorted(os.listdir(args.raw_path))
    assert len(files) == SEED_NUM_SUBJECT

    for a_file in files:
        assert os.path.exists(os.path.join(args.feature_path, a_file))

    channel_map = {}

    if args.data_name == 'SEED':
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

    grid_mat = np.zeros(shape=(9, 9))
    for i in range(62):
        grid_mat[channel_map[i][0], channel_map[i][1]] = 1
    plt.imshow(grid_mat)
    plt.show()

    # Enumerate all files
    for a_file in tqdm(files):
        raw_dict = sio.loadmat(os.path.join(args.raw_path, a_file))
        feature_dict = sio.loadmat(os.path.join(args.raw_path, a_file))

        print(raw_dict.keys())
        print(feature_dict.keys())

        raw_data = []
        feature_data = []

        for key in raw_dict.keys():
            if 'eeg' in key:
                current_data = raw_dict[key]
                current_data = current_data[:, :-1]  # Exclude the last point

                current_data = (current_data - current_data.mean(axis=0)) / current_data.std(axis=0)

                points = np.array([value[0] * 9 + value[1] for value in channel_map.values()])
                values = np.zeros((9, 9))
                np.put(values, points, current_data[:, 0])

                ############### Only for test ###############
                plt.imshow(values)
                plt.title('original_values')
                plt.colorbar()
                plt.show()
                ############### Only for test ###############

                #             grid_x, grid_y = np.meshgrid(np.arange(9), np.arange(9))
                #             interpolated_mat = interpolate.griddata(points, values, (grid_x, grid_y), method='cubic')
                #             interpolator = interpolate.interp2d(x=points[:,1], y=points[:,0], z=values, kind='cubic')
                interpolator = interpolate.interp2d(x=np.arange(9), y=np.arange(9), z=values, kind='cubic')
                interpolated_mat = interpolator(np.linspace(0, 8, 32), np.linspace(0, 8, 32))

                ############### Only for test ###############
                print(points.shape)
                print(interpolated_mat)
                plt.imshow(interpolated_mat)
                plt.title('interpolated_mat')
                plt.colorbar()
                plt.show()
                ############### Only for test ###############

                raw_data.append(current_data)
                print(key, current_data.shape)

        for key in feature_dict.keys():
            if key.startswith('de_LDS'):
                current_data = feature_dict[key]
                #             assert current_data.shape[:2] == raw_data[len(feature_data)].shape[:2], f'{key}: {current_data.shape[:2]} - {raw_data[i].shape[:2]}'

                current_data = (current_data - current_data.mean(axis=1)[:, np.newaxis, :]) / current_data.std(axis=1)[
                                                                                              :, np.newaxis, :]
                points = np.array([value[0] * 9 + value[1] for value in channel_map.values()])
                values = np.zeros((9, 9))
                np.put(values, points, current_data[:, 0, 0])

                ############### Only for test ###############
                plt.imshow(values)
                plt.title('original_values')
                plt.colorbar()
                plt.show()
                ############### Only for test ###############

                interpolator = interpolate.interp2d(x=np.arange(9), y=np.arange(9), z=values, kind='cubic')
                interpolated_mat = interpolator(np.linspace(0, 8, 32), np.linspace(0, 8, 32))

                ############### Only for test ###############
                print(points.shape)
                print(interpolated_mat)
                plt.imshow(interpolated_mat)
                plt.title('interpolated_mat')
                plt.colorbar()
                plt.show()
                ############### Only for test ###############

                feature_data.append(current_data)
                print(key, feature_dict[key].shape)
