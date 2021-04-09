"""
@Time    : 2021/4/9 15:36
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dpc.py
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import KFold
from tqdm.std import tqdm

from mme import Encoder
from mme import adjust_learning_rate, logits_accuracy, mask_accuracy, get_performance
from mme import SEEDDataset, DEAPDataset, AMIGOSDataset
from mme.dataset import SEED_NUM_SUBJECT, DEAP_NUM_SUBJECT, AMIGOS_NUM_SUBJECT


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
    parser.add_argument('--pred-steps', type=int, default=5)

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


class DPC(nn.Module):
    def __init__(self, input_size, input_channels, feature_dim, pred_steps, use_temperature, temperature,
                 device):
        super(DPC, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.device = device

        self.encoder = Encoder(input_size, input_channels, feature_dim)
        self.agg = nn.GRU(input_size=feature_dim, hidden_size=feature_dim, batch_first=True)
        self.predictors = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for i in range(pred_steps)])

        self.relu = nn.ReLU(inplace=True)
        self.targets = None

        # self._initialize_weights(self.agg)
        # self._initialize_weights(self.predictor)

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)

        feature = feature.view(batch_size, num_epoch, self.feature_dim)
        feature_relu = self.relu(feature)

        ### aggregate, predict future ###
        _, hidden = self.agg(feature_relu[:, 0:num_epoch - self.pred_steps, :].contiguous())
        hidden = hidden[:, -1, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        pred = []
        for i in range(self.pred_steps):
            # sequentially pred future
            p_tmp = self.predictor(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)

        # Feature: (batch_size, num_epoch, feature_size, last_size)
        # Pred: (batch_size, pred_steps, feature_size, last_size)
        feature = feature.permute(0, 1, 3, 2).contiguous()
        feature = F.normalize(feature, p=2, dim=-1)

        pred = pred.permute(0, 1, 3, 2).contiguous()
        pred = F.normalize(pred, p=2, dim=-1)

        logits = torch.einsum('ijkl,mnql->ijkqnm', [feature, pred])
        # print('3. Logits: ', logits.shape)
        logits = logits.view(batch_size * num_epoch, self.pred_steps * batch_size)
        if self.use_temperature:
            logits /= self.temperature

        if self.targets is None:
            targets = torch.zeros(batch_size, num_epoch, self.pred_steps, batch_size)
            for i in range(batch_size):
                for j in range(self.pred_steps):
                    targets[i, num_epoch - self.pred_steps + j, j, i] = 1
            targets = targets.view(batch_size * num_epoch, self.pred_steps * batch_size)
            targets = targets.argmax(dim=1)
            targets = targets.cuda(device=self.device)
            self.targets = targets

        return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class DPCClassifier(nn.Module):
    def __init__(self, input_size, input_channels, feature_dim, pred_steps, num_class,
                 use_l2_norm, use_dropout, use_batch_norm, device):
        super(DPCClassifier, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.device = device
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm

        self.encoder = Encoder(input_size, input_channels, feature_dim)
        # self.agg = nn.GRU(input_size=feature_dim, hidden_size=feature_dim, batch_first=True)

        # self.relu = nn.ReLU(inplace=True)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(self.feature_size))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_dim, num_class))
        self.final_fc = nn.Sequential(*final_fc)

        # self._initialize_weights(self.final_fc)

    def forward(self, x):
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)

        # feature = feature.view(batch_size, num_epoch, self.feature_dim)

        # print('2. Context: ', context.shape)

        if self.use_l2_norm:
            feature = F.normalize(feature, p=2, dim=-1)

        out = self.final_fc(feature)

        # print('3. Out: ', out.shape)

        return out


def pretrain(run_id, model, dataset, device, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
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

                output, target = model(x)

                loss = criterion(output, target)
                acc = logits_accuracy(output, target, topk=(1,))[0]
                accuracies.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})


def finetune(classifier, dataset, device, args):
    params = []
    if args.finetune_mode == 'freeze':
        print('[INFO] Finetune classifier only for the last layer...')
        for name, param in classifier.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})
    elif args.finetune_mode == 'smaller':
        print('[INFO] Finetune the whole classifier where the backbone have a smaller lr...')
        for name, param in classifier.named_parameters():
            if 'encoder' in name:
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})
    else:
        print('[INFO] Finetune the whole classifier...')
        for name, param in classifier.named_parameters():
            params.append({'params': param})

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
    else:
        raise ValueError('Invalid optimizer!')

    criterion = nn.CrossEntropyLoss().cuda(device)

    sampled_indices = np.arange(len(dataset))
    np.random.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:int(len(sampled_indices) * args.finetune_ratio)]
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True, drop_last=True,
                             sampler=SubsetRandomSampler(sampled_indices))

    classifier.train()
    for epoch in range(args.finetune_epochs):
        losses = []
        accuracies = []
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.finetune_epochs}]') as progress_bar:
            for x, y in progress_bar:
                x, y = x.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True)

                out = classifier(x)
                loss = criterion(out.view(args.batch_size * args.num_seq, -1), y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accuracies.append(
                    logits_accuracy(out.view(args.batch_size * args.num_seq, -1), y.view(-1), topk=(1,))[0])

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})


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

    model = CPC(input_size=input_size, input_channels=args.input_channel, feature_dim=args.feature_dim,
                pred_steps=args.pred_steps, use_temperature=True, temperature=0.07, device=args.device)
    model.cuda(args.device)

    train_dataset = eval(f'{args.data_name}Dataset')(args.data_path, args.num_seq, train_patients,
                                                     label_dim=args.label_dim)
    pretrain(run_id, model, train_dataset, args.device, args)

    # Finetuning
    if args.finetune_mode == 'freeze':
        use_dropout = False
        use_l2_norm = True
        use_final_bn = True
    else:
        use_dropout = True
        use_l2_norm = False
        use_final_bn = False

    classifier = CPCClassifier(input_size=input_size, input_channels=args.input_channel, feature_dim=args.feature_dim,
                               num_class=args.classes, use_dropout=use_dropout, use_l2_norm=use_l2_norm,
                               use_batch_norm=use_final_bn, device=args.device)
    classifier.cuda(args.device)

    classifier.load_state_dict(model.state_dict(), strict=False)

    finetune(classifier, train_dataset, args.device, args)


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
