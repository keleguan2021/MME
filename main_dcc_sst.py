"""
@Time    : 2021/4/1 20:20
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main_dcc_sst.py
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
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.std import tqdm

from mme import DCC, DCCClassifier
from mme import (
    adjust_learning_rate, logits_accuracy, mask_accuracy, get_performance
)
from mme import SEEDDataset, SEEDIVDataset, DEAPDataset, AMIGOSDataset, SEEDSSTDataset, SEEDIVSSTDataset
from mme.data import SEED_NUM_SUBJECT, SEED_IV_NUM_SUBJECT, DEAP_NUM_SUBJECT, AMIGOS_NUM_SUBJECT


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
    parser.add_argument('--data-path', type=str, default='data/sst_feature/SEED/raw')
    parser.add_argument('--data-name', type=str, default='SEED', choices=['SEED', 'SEED-IV', 'DEAP', 'AMIGOS'])
    parser.add_argument('--save-path', type=str, default='./cache/tmp')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--label-dim', type=int, default=0, help='Ignored for SEED')
    parser.add_argument('--feature-mode', type=str, default='raw', choices=['raw', 'de'])

    # Model
    parser.add_argument('--input-channel', type=int, default=62)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-seq', type=int, default=10)
    parser.add_argument('--grid-res', type=int, default=16)

    # Training
    parser.add_argument('--only-pretrain', action='store_true')
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--finetune-mode', type=str, default='freeze', choices=['freeze', 'smaller', 'all'])

    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

    # Distributed training
    parser.add_argument('--use-dist', action='store_true')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:12345', type=str,
                        help='url used to set up distributed training')

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
    else:
        raise ValueError('Invalid optimizer!')

    criterion = nn.CrossEntropyLoss().cuda(device)

    if args.use_dist:
        sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=(sampler is None), pin_memory=True, drop_last=True, sampler=sampler)
    else:
        data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=True, pin_memory=True, drop_last=True)

    model.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        accuracies = []
        if args.use_dist:
            data_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, args.lr, epoch, args.pretrain_epochs, args)
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.pretrain_epochs}]') as progress_bar:
            for x, _ in progress_bar:
                x = x.cuda(device, non_blocking=True)

                # output, target = model(x)

                # loss = criterion(output, target)

                loss = model(x)

                # acc = logits_accuracy(output, target, topk=(1,))[0]
                # accuracies.append(acc)

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


def run(gpu, ngpus_per_node, run_id, train_patients, test_patients, args):
    if args.use_dist:
        print(f'[INFO] Process ({gpu}) invoked among {ngpus_per_node} gpus...')

    # Unique random seeds for each thread
    if args.seed is not None:
        setup_seed(args.seed + gpu)

    if args.use_dist:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node,
                                rank=gpu)

    if gpu == 0:
        print('Train patient ids:', train_patients)
        print('Test patient ids:', test_patients)

    if args.data_name == 'SEED' or args.data_name == 'SEED-IV':
        input_size = 200
    elif args.data_name == 'DEAP':
        input_size = 128
    elif args.data_name == 'AMIGOS':
        input_size = 128
    else:
        raise ValueError

    if args.use_dist:
        torch.cuda.set_device(gpu)

    if args.feature_mode == 'raw':
        model = DCC((200, args.grid_res, args.grid_res), 1, args.feature_dim, True, 0.07,
                    gpu if args.use_dist else args.device, mode='sst', strides=(1, 2, 2, 2), use_dist=args.use_dist)
    else:
        model = DCC((5, args.grid_res, args.grid_res), 1, args.feature_dim, True, 0.07,
                    gpu if args.use_dist else args.device, mode='sst', strides=(1, 1, 2, 2), use_dist=args.use_dist)
    if args.use_dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        model_without_ddp = model.module
    else:
        model.cuda(args.device)

    if args.data_name == 'SEED':
        train_dataset = SEEDSSTDataset(args.data_path, args.num_seq, train_patients, label_dim=args.label_dim)
    elif args.data_name == 'SEED-IV':
        train_dataset = SEEDIVSSTDataset(args.data_path, args.num_seq, train_patients, label_dim=args.label_dim)
    elif args.data_name == 'DEAP':
        train_dataset = DEAPSSTDataset(args.data_path, args.num_seq, train_patients, label_dim=args.label_dim)
    else:
        raise ValueError

    pretrain(run_id, model, train_dataset, gpu if args.use_dist else args.device, args)
    if args.use_dist:
        torch.save(model.module.state_dict(),
                   os.path.join(args.save_path, f'dcc_{args.feature_mode}_pretrained.pth.tar'))
    else:
        torch.save(model.state_dict(),
                   os.path.join(args.save_path, f'dcc_{args.feature_mode}_pretrained.pth.tar'))

    if not args.only_pretrain and gpu == 0:
        print('[INFO] Start finetuning on the first gpu...')

        # Finetuning
        if args.finetune_mode == 'freeze':
            use_dropout = False
            use_l2_norm = True
            use_final_bn = True
        else:
            use_dropout = True
            use_l2_norm = False
            use_final_bn = False

        if args.feature_mode == 'raw':
            classifier = DCCClassifier(input_size=(200, args.grid_res, args.grid_res), input_channels=1,
                                       feature_dim=args.feature_dim,
                                       num_class=args.classes,
                                       use_dropout=use_dropout, use_l2_norm=use_l2_norm, use_batch_norm=use_final_bn,
                                       device=gpu if args.use_dist else args.device, mode='sst', strides=(1, 2, 2, 2))
        else:
            classifier = DCCClassifier(input_size=(5, args.grid_res, args.grid_res), input_channels=1,
                                       feature_dim=args.feature_dim,
                                       num_class=args.classes,
                                       use_dropout=use_dropout, use_l2_norm=use_l2_norm, use_batch_norm=use_final_bn,
                                       device=gpu if args.use_dist else args.device, mode='sst', strides=(1, 1, 2, 2))

        classifier.cuda(gpu)

        if args.use_dist:
            classifier.load_state_dict(model.module.state_dict(), strict=False)
        else:
            classifier.load_state_dict(model.state_dict(), strict=False)

        finetune(classifier, train_dataset, gpu if args.use_dist else args.device, args)

        if args.data_name == 'SEED':
            test_dataset = SEEDSSTDataset(args.data_path, args.num_seq, test_patients, label_dim=args.label_dim)
        elif args.data_name == 'SEED-IV':
            test_dataset = SEEDIVSSTDataset(args.data_path, args.num_seq, test_patients, label_dim=args.label_dim)
        # elif args.data_name == 'DEAP':
        #     test_dataset = DEAPDataset(args.data_path, args.num_seq, test_patients, label_dim=args.label_dim)
        else:
            raise ValueError

        # test_dataset = eval(f'{args.data_name}Dataset')(args.data_path, args.num_seq, test_patients,
        #                                                    label_dim=args.label_dim)
        scores, targets = evaluate(classifier, test_dataset, gpu if args.use_dist else args.device, args)
        performance = get_performance(scores, targets)
        with open(os.path.join(args.save_path, f'statistics_{run_id}.pkl'), 'wb') as f:
            pickle.dump({'performance': performance, 'args': vars(args)}, f)
        print(performance)


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.save_path):
        warnings.warn(f'The path {args.save_path} dost not existed, created...')
        os.makedirs(args.save_path)
    elif not args.resume:
        warnings.warn(f'The path {args.save_path} already exists, deleted...')
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    if args.data_name == 'SEED':
        num_patients = SEED_NUM_SUBJECT
    elif args.data_name == 'SEED-IV':
        num_patients = SEED_IV_NUM_SUBJECT
    elif args.data_name == 'DEAP':
        num_patients = DEAP_NUM_SUBJECT
    elif args.data_name == 'AMIGOS':
        num_patients = AMIGOS_NUM_SUBJECT
    else:
        raise ValueError

    patients = np.arange(num_patients)

    ngpus_per_node = torch.cuda.device_count()

    assert args.kfold <= len(patients)
    assert args.fold < args.kfold
    kf = KFold(n_splits=args.kfold)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        if i == args.fold:
            print(f'[INFO] Running cross validation for {i + 1}/{args.kfold} fold...')
            train_patients, test_patients = patients[train_index].tolist(), patients[test_index].tolist()
            if args.use_dist:
                mp.spawn(run, nprocs=ngpus_per_node, args=(ngpus_per_node, i, train_patients, test_patients, args))
            else:
                run(0, None, i, train_patients, test_patients, args)
            break
