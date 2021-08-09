"""
@Time    : 2021/6/16 21:05
@File    : eval_dcc.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import pickle
import random
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.std import tqdm

from mme import DCC, DCCClassifier
from mme import adjust_learning_rate, logits_accuracy, mask_accuracy, get_performance
from mme import SEEDDataset, SEEDIVDataset, DEAPDataset, AMIGOSDataset, SleepDataset, SleepDatasetImg
from mme.data import SEED_NUM_SUBJECT, SEED_IV_NUM_SUBJECT, DEAP_NUM_SUBJECT, AMIGOS_NUM_SUBJECT, \
    ISRUC_NUM_SUBJECT, SLEEPEDF_NUM_SUBJECT


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
    parser.add_argument('--save-path', type=str, default='./cache/tmp')
    parser.add_argument('--load-path', type=str, required=True)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--label-dim', type=int, default=0, help='Ignored for SEED')
    parser.add_argument('--preprocessing', choices=['none', 'quantile', 'standard'], default='standard')

    # Model
    parser.add_argument('--input-channel', type=int, default=62)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-seq', type=int, default=10)

    # Training
    # parser.add_argument('--pretrain-epochs', type=int, default=200)
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
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
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


def evaluate(classifier, dataset, device, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    targets = []
    scores = []

    classifier.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda(device, non_blocking=True)

            out = classifier(x)
            scores.append(out.view(args.batch_size * args.num_seq, -1).cpu().numpy())
            targets.append(y.view(-1).numpy())

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    return scores, targets


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
        train_dataset = SEEDDataset(args.data_path, args.num_seq, train_patients, label_dim=args.label_dim)
    elif args.data_name == 'SEED-IV':
        train_dataset = SEEDIVDataset(args.data_path, args.num_seq, train_patients, label_dim=args.label_dim)
    elif args.data_name == 'DEAP':
        train_dataset = DEAPDataset(args.data_path, args.num_seq, train_patients, label_dim=args.label_dim)
    elif args.data_name == 'AMIGOS':
        train_dataset = AMIGOSDataset(args.data_path, args.num_seq, train_patients, label_dim=args.label_dim)
    elif args.data_name == 'ISRUC':
        train_dataset = SleepDataset(args.data_path, 'isruc', args.num_seq, train_patients,
                                     preprocessing=args.preprocessing)
    elif args.data_name == 'SLEEPEDF':
        train_dataset = SleepDataset(args.data_path, 'sleepedf', args.num_seq, train_patients,
                                     preprocessing=args.preprocessing)
    else:
        raise ValueError

    # Finetuning
    if args.finetune_mode == 'freeze':
        use_dropout = False
        use_l2_norm = True
        use_final_bn = True
    else:
        use_dropout = True
        use_l2_norm = False
        use_final_bn = False

    classifier = DCCClassifier(input_size=input_size, input_channels=args.input_channel, feature_dim=args.feature_dim,
                               num_class=args.classes,
                               use_dropout=use_dropout, use_l2_norm=use_l2_norm, use_batch_norm=use_final_bn,
                               device=args.device)
    classifier.cuda(args.device)

    classifier.load_state_dict(torch.load(args.load_path), strict=False)

    print('[INFO] Start fine-tuning...')
    finetune(classifier, train_dataset, args.device, args)

    if args.data_name == 'SEED':
        test_dataset = SEEDDataset(args.data_path, args.num_seq, test_patients, label_dim=args.label_dim)
    elif args.data_name == 'SEED-IV':
        test_dataset = SEEDIVDataset(args.data_path, args.num_seq, test_patients, label_dim=args.label_dim)
    elif args.data_name == 'DEAP':
        test_dataset = DEAPDataset(args.data_path, args.num_seq, test_patients, label_dim=args.label_dim)
    elif args.data_name == 'AMIGOS':
        test_dataset = AMIGOSDataset(args.data_path, args.num_seq, test_patients, label_dim=args.label_dim)
    elif args.data_name == 'ISRUC':
        test_dataset = SleepDataset(args.data_path, 'isruc', args.num_seq, test_patients,
                                    preprocessing=args.preprocessing)
    elif args.data_name == 'SLEEPEDF':
        test_dataset = SleepDataset(args.data_path, 'sleepedf', args.num_seq, test_patients,
                                    preprocessing=args.preprocessing)
    else:
        raise ValueError

    scores, targets = evaluate(classifier, test_dataset, args.device, args)
    performance = get_performance(scores, targets)
    with open(os.path.join(args.save_path, f'statistics_{run_id}.pkl'), 'wb') as f:
        pickle.dump({'performance': performance, 'args': vars(args)}, f)
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
        patients = np.arange(SEED_NUM_SUBJECT)
    elif args.data_name == 'SEED-IV':
        patients = np.arange(SEED_IV_NUM_SUBJECT)
    elif args.data_name == 'DEAP':
        patients = np.arange(DEAP_NUM_SUBJECT)
    elif args.data_name == 'AMIGOS':
        patients = np.arange(AMIGOS_NUM_SUBJECT)
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
