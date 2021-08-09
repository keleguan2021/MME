"""
@Time    : 2021/3/22 23:33
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : tnc.py
@Software: PyCharm
@Desc    : 
"""
import os
import sys
import shutil
import math
import random
import warnings
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import KFold
from tqdm.std import tqdm

from mme import Encoder
from mme.data import SEED_NUM_SUBJECT, DEAP_NUM_SUBJECT, AMIGOS_NUM_SUBJECT


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
    parser.add_argument('--data-name', type=str, default='SEED', choices=['SEED', 'DEAP', 'AMIGOS'])
    parser.add_argument('--save-path', type=str, default='./cache/tmp')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--label-dim', type=int, default=0, help='Ignored for SEED')

    # Model
    parser.add_argument('--input-channel', type=int, default=62)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-seq', type=int, default=10)

    # Training
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--finetune-mode', type=str, default='freeze', choices=['freeze', 'smaller', 'all'])

    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

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


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size

        self.model = torch.nn.Sequential(torch.nn.Linear(2 * self.input_size, 4 * self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4 * self.input_size, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))


class TNCDataset(Dataset):
    def __init__(self, x, mc_sample_size, window_size, augmentation, epsilon=3, state=None, adf=False):
        super(TNCDataset, self).__init__()
        self.time_series = x  # (num_sample, channel, time_len)
        self.T = x.shape[-1]
        self.window_size = window_size
        self.sliding_gap = int(window_size * 25.2)
        self.window_per_sample = (self.T - 2 * self.window_size) // self.sliding_gap
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.augmentation = augmentation
        self.adf = adf
        if not self.adf:
            self.epsilon = epsilon
            self.delta = 5 * window_size * epsilon

    def __len__(self):
        return len(self.time_series) * self.augmentation

    def __getitem__(self, ind):
        ind = ind % len(self.time_series)
        t = np.random.randint(2 * self.window_size, self.T - 2 * self.window_size)
        x_t = self.time_series[ind][:, t - self.window_size // 2:t + self.window_size // 2]
        # plt.savefig('./plots/%s_seasonal.png'%ind)
        X_close = self._find_neighours(self.time_series[ind], t)
        X_distant = self._find_non_neighours(self.time_series[ind], t)

        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[ind][t - self.window_size // 2:t + self.window_size // 2]))
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size, 4 * self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0, t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val / x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0]) == 0 else (
                        np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5 * self.epsilon * self.window_size

        ## Random from a Gaussian
        t_p = [int(t + np.random.randn() * self.epsilon * self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size // 2 + 1, min(t_pp, T - self.window_size // 2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind - self.window_size // 2:t_ind + self.window_size // 2] for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t > T / 2:
            t_n = np.random.randint(self.window_size // 2, max((t - self.delta + 1), self.window_size // 2 + 1),
                                    self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size - 1)), (T - self.window_size // 2),
                                    self.mc_sample_size)
        x_n = torch.stack([x[:, t_ind - self.window_size // 2:t_ind + self.window_size // 2] for t_ind in t_n])

        if len(x_n) == 0:
            rand_t = np.random.randint(0, self.window_size // 5)
            if t > T / 2:
                x_n = x[:, rand_t:rand_t + self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n


def pretrain(run_id, encoder, discriminator, dataset, device, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(list(encoder.parameters()) + list(discriminator.parameters()), lr=args.lr,
                              weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(list(encoder.parameters()) + list(discriminator.parameters()), lr=args.lr,
                               weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
    else:
        raise ValueError('Invalid optimizer!')

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    encoder.train()
    discriminator.train()
    for epoch in range(args.pretrain_epochs):
        pass


def finetune():
    pass


def evaluate():
    pass


def run(run_id, train_patients, test_patients, args):
    print('Train patient ids:', train_patients)
    print('Test patient ids:', test_patients)

    if args.data_name == 'SEED':
        input_size = 200
    elif args.data_name == 'DEAP':
        input_size = 128
    elif args.data_name == 'AMIGOS':
        input_size = 128
    else:
        raise ValueError

    encoder = Encoder(input_size=input_size, input_channel=args.input_channel, feature_dim=args.feature_dim)
    encoder.cuda(args.device)

    discriminator = Discriminator(args.feature_dim, args.device)
    discriminator.cuda(args.device)

    dataset = TNCDataset()

    pretrain(run_id, encoder, discriminator, dataset, args.device, args)


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
    elif args.data_name == 'DEAP':
        num_patients = DEAP_NUM_SUBJECT
    elif args.data_name == 'AMIGOS':
        num_patients = AMIGOS_NUM_SUBJECT
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
