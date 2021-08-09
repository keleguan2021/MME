"""
@Time    : 2021/2/4 12:22
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : train_dcc.py
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
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm.std import tqdm

from mme import DCC
from mme import adjust_learning_rate
from mme.data.datasets import SEEDDataset, SEEDIVDataset, DEAPDataset, AMIGOSDataset, SleepEDFDataset, ISRUCDataset


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
    parser.add_argument('--data-name', type=str, default='SEED',
                        choices=['SEED', 'SEED-IV', 'DEAP', 'AMIGOS', 'ISRUC', 'SLEEPEDF'])
    parser.add_argument('--modal', type=str, default='eeg', choices=['eeg', 'eog', 'emg'])
    parser.add_argument('--save-path', type=str, default='./cache/tmp')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--label-dim', type=int, default=0, help='Ignored for SEED')
    parser.add_argument('--preprocessing', choices=['none', 'quantile', 'standard'], default='standard')

    # Model
    parser.add_argument('--input-channel', type=int, default=None)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-seq', type=int, default=10)
    # parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'sst', 'img'])

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
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

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


def pretrain(run_id, model, dataset, device, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    else:
        raise ValueError('Invalid optimizer!')

    criterion = nn.CrossEntropyLoss().cuda(device)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    model.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        accuracies = []
        adjust_learning_rate(optimizer, args.lr, epoch, args.pretrain_epochs, args)
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.pretrain_epochs}]') as progress_bar:
            for x, _ in progress_bar:
                x = x.cuda(device, non_blocking=True)

                loss = model(x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})


def main_worker(run_id, train_patients, test_patients, args):
    print('Train patient ids:', train_patients)
    print('Test patient ids:', test_patients)

    if args.data_name == 'SEED' or args.data_name == 'SEED-IV':
        input_size = 200
    elif args.data_name == 'DEAP':
        input_size = 128
    elif args.data_name == 'AMIGOS':
        input_size = 128
    elif args.data_name == 'ISRUC':
        input_size = 200
    elif args.data_name == 'SLEEPEDF':
        input_size = 100
    else:
        raise ValueError

    if args.data_name == 'SEED':
        train_dataset = SEEDDataset(data_path=args.data_path, num_seq=args.num_seq,
                                    subject_list=train_patients, label_dim=args.label_dim)
    elif args.data_name == 'SEED-IV':
        train_dataset = SEEDIVDataset(data_path=args.data_path, num_seq=args.num_seq,
                                      subject_list=train_patients, label_dim=args.label_dim)
    elif args.data_name == 'DEAP':
        train_dataset = DEAPDataset(data_path=args.data_path, num_seq=args.num_seq,
                                    subject_list=train_patients, label_dim=args.label_dim, modal=args.modal)
    # elif args.data_name == 'AMIGOS':
    #     train_dataset = AMIGOSDataset(args.data_path, args.num_seq, train_patients, label_dim=args.label_dim)
    elif args.data_name == 'ISRUC':
        train_dataset = ISRUCDataset(data_path=args.data_path, num_epoch=args.num_seq, patients=train_patients,
                                     modal=args.modal)
    elif args.data_name == 'SLEEPEDF':
        train_dataset = SleepEDFDataset(data_path=args.data_path, num_epoch=args.num_seq, patients=train_patients,
                                        modal=args.modal)
    else:
        raise ValueError

    model = DCC(input_size, args.input_channel if args.input_channel is not None else train_dataset.channels,
                args.feature_dim, False, 0.07, args.device)
    model.cuda(args.device)

    print('[INFO] Start pretraining...')
    pretrain(run_id, model, train_dataset, args.device, args)
    torch.save(model.state_dict(),
               os.path.join(args.save_path, f'dcc_{args.data_name}_{args.modal}_{run_id}_pretrain_final.pth.tar'))


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
        patients = np.arange(SEEDDataset.num_subject)
    elif args.data_name == 'SEED-IV':
        patients = np.arange(SEEDIVDataset.num_subject)
    elif args.data_name == 'DEAP':
        patients = np.arange(DEAPDataset.num_subject)
    elif args.data_name == 'AMIGOS':
        patients = np.arange(AMIGOSDataset.num_subject)
    elif args.data_name == 'ISRUC' or args.data_name == 'SLEEPEDF':
        files = os.listdir(args.data_path)
        patients = []
        for a_file in files:
            if a_file.endswith('.npz'):
                patients.append(a_file)

        patients = sorted(patients)
        patients = np.asarray(patients)
    else:
        raise ValueError

    assert args.kfold <= len(patients)
    assert args.fold < args.kfold
    kf = KFold(n_splits=args.kfold)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        if i == args.fold:
            print(f'[INFO] Running cross validation for {i + 1}/{args.kfold} fold...')
            train_patients, test_patients = patients[train_index].tolist(), patients[test_index].tolist()
            main_worker(i, train_patients, test_patients, args)
            break
