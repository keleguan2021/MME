"""
@Time    : 2021/2/4 12:22
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import random
import shutil
import warnings

import numpy as np
import torch
from sklearn.model_selection import KFold

from mme import DCC
from mme.dataset import SEEDDataset, SEED_NUM_SUBJECT


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data-path', type=str, default='/data/DataHub/EmotionRecognition/SEED/Preprocessed_EEG')
    parser.add_argument('--data-name', type=str, default='SEED')
    parser.add_argument('--save-path', type=str, default='./cache/tmp')

    # Model
    parser.add_argument('--input-channel', type=int, default=62)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-seq', type=int, default=10)

    # Training
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--num-patients', type=int)

    parser.add_argument('--seed', type=int, default=2020)

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


def pretrain():
    pass


def finetune():
    pass


def evaluate():
    pass


def run(run_id, train_patients, test_patients, args):
    print('Train patient ids:', train_patients)
    print('Test patient ids:', test_patients)

    model = DCC(args.input_channel, args.feature_dim, True, 0.07)
    model.cuda(args.device)

    train_dataset = SEEDDataset(args.data_path, args.num_seq, train_patients)
    pretrain()


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    if not os.path.exists(args.save_path):
        warnings.warn(f'The path {args.save_path} dost not existed, created...')
        os.makedirs(args.save_path)
    elif not args.resume:
        warnings.warn(f'The path {args.save_path} already exists, deleted...')
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    if args.data_name == 'SEED':
        num_patients = SEED_NUM_SUBJECT
    else:
        raise ValueError

    patients = np.arange(num_patients)

    assert args.kfold <= len(patients)
    assert args.fold < args.kfold
    kf = KFold(n_splits=args.kfold)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        if i == args.fold:
            print(f'[INFO] Running cross validation for {i + 1}/{args.kfold} fold...')
            train_patients, test_patients = patients[train_index].tolist(), patients[test_index].tolist()
            run(i, train_patients, test_patients, args)
            break
