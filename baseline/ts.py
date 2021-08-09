"""
@Time    : 2021/3/22 13:37
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : ts.py
@Software: PyCharm
@Desc    : 
"""
import os
import sys
import pickle
import shutil
import random
import warnings
import argparse
from typing import List

import numpy as np
import scipy.io as sio
from tqdm.std import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from mme import Encoder
from mme import logits_accuracy, get_performance
from mme import SEEDDataset, DEAPDataset, AMIGOSDataset
from mme.data import (
    SEED_NUM_SUBJECT, SEED_SAMPLING_RATE, SEED_LABELS,
    DEAP_NUM_SUBJECT, DEAP_SAMPLING_RATE,
    AMIGOS_NUM_SUBJECT, AMIGOS_SAMPLING_RATE
)


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

    # Dataset & saving & loading
    parser.add_argument('--data-path', type=str, default='/data/DataHub/EmotionRecognition/SEED/Preprocessed_EEG')
    parser.add_argument('--data-name', type=str, default='SEED', choices=['SEED', 'DEAP', 'AMIGOS'])
    parser.add_argument('--save-path', type=str, default='./cache/tmp')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--label-dim', type=int, default=0, help='Ignored for SEED')

    # Model
    parser.add_argument('--input-channel', type=int, default=62)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-seq', type=int, default=10)
    parser.add_argument('--num-sampling', type=int, default=5000)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--pred-steps', type=int, default=5)
    parser.add_argument('--dis', type=int, default=10)

    # Training
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--finetune-mode', type=str, default='freeze', choices=['freeze', 'smaller', 'all'])
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--use-temperature', action='store_true')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

    # Misc
    parser.add_argument('--disp-interval', type=int, default=20)
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


class SoftLogitLoss(nn.Module):
    def __init__(self):
        super(SoftLogitLoss, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # print(output.shape, target.shape)
        if len(target.shape) == 1:
            target = target.view(-1, 1)
        loss = torch.log(1 + torch.exp(-target * output)).mean()

        return loss


class TSDataset(Dataset):
    def __init__(self, data_path, data_name, num_sampling, dis, patients: List = None, verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.patients = patients
        self.num_sampling = num_sampling
        self.dis = dis

        files = sorted(os.listdir(data_path))
        files = [files[i] for i in patients]

        all_data = []
        all_label = []

        # Enumerate all files
        for a_file in tqdm(files):
            # print(f'[INFO] Processing file {a_file}...')
            data = sio.loadmat(os.path.join(data_path, a_file))

            if data_name == 'SEED':
                # Each file contains 15 consecutive trials
                movie_ids = list(filter(lambda x: not x.startswith('__'), data.keys()))
                subject_data = []
                subject_label = []
                assert len(movie_ids) == len(SEED_LABELS)

                # Enumerate all trials
                for key in movie_ids:
                    trial_data = data[key]
                    trial_data = trial_data[:, :-1]  # remove the last redundant point
                    # trial_data = tensor_standardize(trial_data, dim=-1)
                    assert trial_data.shape[1] % SEED_SAMPLING_RATE == 0
                    trial_data = trial_data.reshape(trial_data.shape[0], trial_data.shape[1] // SEED_SAMPLING_RATE,
                                                    SEED_SAMPLING_RATE)
                    trial_data = np.swapaxes(trial_data, 0, 1)

                    num_to_sample_trial = num_sampling // len(files) // len(movie_ids)
                    anchor_indices = np.random.choice(np.arange(0, trial_data.shape[0] - dis),
                                                      size=num_to_sample_trial // 2,
                                                      replace=False)
                    trial_selected = []
                    label_selected = []
                    for k in anchor_indices:
                        pos_idx = np.random.randint(k + 1, k + dis)
                        neg_idx = np.ones(trial_data.shape[0], dtype=np.bool)
                        neg_idx[np.arange(k, k + dis + 1)] = False
                        neg_idx = np.arange(trial_data.shape[0])[neg_idx]
                        neg_idx = np.random.choice(neg_idx, size=1)[0]
                        # self.data.append(np.stack([recordings[k], recordings[pos_idx], recordings[k + dis]], axis=0))
                        trial_selected.append(trial_data[[k, pos_idx, k + dis]])
                        label_selected.append(1)
                        # self.data.append(np.stack([recordings[k], recordings[neg_idx], recordings[k + dis]], axis=0))
                        trial_selected.append(trial_data[[k, neg_idx, k + dis]])
                        label_selected.append(-1)
                    trial_selected = np.stack(trial_selected, axis=0)
                    label_selected = np.array(label_selected, dtype=np.long)

                    subject_data.append(trial_selected)
                    subject_label.append(label_selected)
                subject_data = np.concatenate(subject_data, axis=0)
                subject_label = np.concatenate(subject_label)

                all_data.append(subject_data)
                all_label.append(subject_label)
            elif data_name == 'DEAP':
                subject_data = data['data']  # trial x channel x data
                subject_label = data['labels']  # trial x label (valence, arousal, dominance, liking)

                subject_data_selected = []
                subject_label_selected = []

                for i_trial in range(subject_data.shape[0]):
                    trial_data = subject_data[i_trial]
                    trial_label = subject_label[i_trial]

                    trial_data = trial_data.reshape(trial_data.shape[0], trial_data.shape[1] // DEAP_SAMPLING_RATE,
                                                    DEAP_SAMPLING_RATE)
                    trial_data = np.swapaxes(trial_data, 0, 1)

                    num_to_sample_trial = num_sampling // len(files) // subject_data.shape[0]
                    anchor_indices = np.random.choice(np.arange(0, trial_data.shape[0] - dis),
                                                      size=num_to_sample_trial // 2,
                                                      replace=False)
                    trial_selected = []
                    label_selected = []
                    for k in anchor_indices:
                        pos_idx = np.random.randint(k + 1, k + dis)
                        neg_idx = np.ones(trial_data.shape[0], dtype=np.bool)
                        neg_idx[np.arange(k, k + dis + 1)] = False
                        neg_idx = np.arange(trial_data.shape[0])[neg_idx]
                        neg_idx = np.random.choice(neg_idx, size=1)[0]
                        # self.data.append(np.stack([recordings[k], recordings[pos_idx], recordings[k + dis]], axis=0))
                        trial_selected.append(trial_data[[k, pos_idx, k + dis]])
                        label_selected.append(1)
                        # self.data.append(np.stack([recordings[k], recordings[neg_idx], recordings[k + dis]], axis=0))
                        trial_selected.append(trial_data[[k, neg_idx, k + dis]])
                        label_selected.append(-1)
                    trial_selected = np.stack(trial_selected, axis=0)
                    label_selected = np.array(label_selected, dtype=np.long)

                    subject_data_selected.append(trial_selected)
                    subject_label_selected.append(label_selected)
                subject_data_selected = np.concatenate(subject_data_selected, axis=0)
                subject_label_selected = np.concatenate(subject_label_selected)

                all_data.append(subject_data_selected)
                all_label.append(subject_label_selected)
            elif data_name == 'AMIGOS':
                subject_data = []
                subject_label = []
                for i in range(data['joined_data'].shape[1]):
                    trial_data = data['joined_data'][0, i]
                    trial_label = data['labels_selfassessment'][0, i]

                    if np.isnan(trial_data).any() or 0 in trial_data.shape or 0 in trial_label.shape:
                        warnings.warn('Malfunctioned data array, dropped.')
                        continue

                    if trial_data.shape[0] % AMIGOS_SAMPLING_RATE != 0:
                        trial_data = trial_data[:trial_data.shape[0] // AMIGOS_SAMPLING_RATE * AMIGOS_SAMPLING_RATE]
                    trial_data = trial_data.reshape(trial_data.shape[0] // AMIGOS_SAMPLING_RATE, AMIGOS_SAMPLING_RATE,
                                                    trial_data.shape[1])
                    trial_data = np.swapaxes(trial_data, 1, 2)

                    num_to_sample_trial = num_sampling // len(files) // data['joined_data'].shape[1]
                    anchor_indices = np.random.choice(np.arange(0, trial_data.shape[0] - dis),
                                                      size=num_to_sample_trial // 2,
                                                      replace=False)
                    trial_selected = []
                    label_selected = []
                    for k in anchor_indices:
                        pos_idx = np.random.randint(k + 1, k + dis)
                        neg_idx = np.ones(trial_data.shape[0], dtype=np.bool)
                        neg_idx[np.arange(k, k + dis + 1)] = False
                        neg_idx = np.arange(trial_data.shape[0])[neg_idx]
                        neg_idx = np.random.choice(neg_idx, size=1)[0]
                        # self.data.append(np.stack([recordings[k], recordings[pos_idx], recordings[k + dis]], axis=0))
                        trial_selected.append(trial_data[[k, pos_idx, k + dis]])
                        label_selected.append(1)
                        # self.data.append(np.stack([recordings[k], recordings[neg_idx], recordings[k + dis]], axis=0))
                        trial_selected.append(trial_data[[k, neg_idx, k + dis]])
                        label_selected.append(-1)
                    trial_selected = np.stack(trial_selected, axis=0)
                    label_selected = np.array(label_selected, dtype=np.long)

                    subject_data.append(trial_selected)
                    subject_label.append(label_selected)
                subject_data = np.concatenate(subject_data, axis=0)
                subject_label = np.concatenate(subject_label)

                all_data.append(subject_data)
                all_label.append(subject_label)
            else:
                raise ValueError
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label)
        print(all_data.shape, all_label.shape)

        self.data = all_data
        self.label = all_label

    def __getitem__(self, item):
        x = self.data[item].astype(np.float32)
        y = self.label[item].astype(np.long)

        return x, y

    def __len__(self):
        return len(self.data)


class TemporalShuffling(nn.Module):
    def __init__(self, input_size, input_channels, hidden_channels, feature_dim, device='cuda'):
        super(TemporalShuffling, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = Encoder(input_size, input_channels, feature_dim)

        self.linear_head = nn.Linear(2 * feature_dim, 1, bias=True)

    def forward(self, x1, x2, x3):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z3 = self.encoder(x3)

        diff1 = torch.abs(z1 - z2)
        diff2 = torch.abs(z2 - z3)

        out = torch.cat([diff1, diff2], dim=-1)
        out = self.linear_head(out)

        return out


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, input_channels, feature_dim, num_classes,
                 use_l2_norm, use_dropout, use_batch_norm, device='cuda'):
        super(SimpleClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = Encoder(input_size, input_channels, feature_dim)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(feature_dim))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_dim, num_classes))
        self.final_fc = nn.Sequential(*final_fc)

    def forward(self, x):
        feature = self.encoder(x)
        # feature = feature.view(batch_size, num_epoch, self.feature_dim)

        if self.use_l2_norm:
            feature = F.normalize(feature, p=2, dim=1)

        out = self.final_fc(feature)

        return out


def pretrain(model, dataset, device, run_id, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
    else:
        raise ValueError('Invalid optimizer!')

    criterion = SoftLogitLoss().cuda(device)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    model.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        accuracies = []
        # adjust_learning_rate(optimizer, args.lr, epoch, args.pretrain_epochs, args)
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.pretrain_epochs}]') as progress_bar:
            for x, y in progress_bar:
                x = x.cuda(device, non_blocking=True)
                y = y.cuda(device, non_blocking=True)

                out = model(x[:, 0], x[:, 1], x[:, 2])

                # print('------ Out:', out.shape)

                loss = criterion(out, y)

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
            if 'encoder' in name or 'agg' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})
    elif args.finetune_mode == 'smaller':
        print('[INFO] Finetune the whole classifier where the backbone have a smaller lr...')
        for name, param in classifier.named_parameters():
            if 'encoder' in name or 'agg' in name:
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

                x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])
                out = classifier(x)
                loss = criterion(out, y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accuracies.append(
                    logits_accuracy(out, y.view(-1), topk=(1,))[0])

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})


def evaluate(classifier, dataset, device, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    targets = []
    scores = []

    classifier.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda(device, non_blocking=True)

            x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])
            out = classifier(x)
            scores.append(out.cpu().numpy())
            targets.append(y.view(-1).numpy())

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    return scores, targets


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

    model = TemporalShuffling(input_size=input_size, input_channels=args.input_channel, hidden_channels=16,
                              feature_dim=args.feature_dim, device=args.device)
    model.cuda(args.device)

    train_dataset = TSDataset(data_path=args.data_path, data_name=args.data_name, num_sampling=args.num_sampling,
                              dis=args.dis, patients=train_patients)
    pretrain(model, train_dataset, args.device, run_id, args)

    del train_dataset

    train_dataset = eval(f'{args.data_name}Dataset')(args.data_path, args.num_seq, train_patients,
                                                     label_dim=args.label_dim)

    if args.finetune_mode == 'freeze':
        use_dropout = False
        use_l2_norm = False
        use_final_bn = True
    else:
        use_dropout = True
        use_l2_norm = False
        use_final_bn = False

    classifier = SimpleClassifier(input_size=input_size, input_channels=args.input_channel,
                                  feature_dim=args.feature_dim, num_classes=args.classes,
                                  use_dropout=use_dropout, use_l2_norm=use_l2_norm, use_batch_norm=use_final_bn,
                                  device=args.device)
    classifier.cuda(args.device)

    classifier.load_state_dict(model.state_dict(), strict=False)

    # Evaluation
    del train_dataset
    test_dataset = eval(f'{args.data_name}Dataset')(args.data_path, args.num_seq, test_patients,
                                                    label_dim=args.label_dim)
    print(test_dataset)
    scores, targets = evaluate(classifier, test_dataset, args.device, args)
    performance = get_performance(scores, targets)
    with open(os.path.join(args.save_path, f'statistics_{run_id}.pkl'), 'wb') as f:
        pickle.dump({'performance': performance, 'args': vars(args), 'cmd': sys.argv}, f)
    print(performance)


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
