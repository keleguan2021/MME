"""
@Time    : 2021/2/6 15:24
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : model.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Encoder, Encoder3d, ResNet2d3d, R1DNet, ResNet


# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#                       for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#
#     output = torch.cat(tensors_gather, dim=0)
#     return output


class DCC(nn.Module):
    def __init__(self, input_size, input_channels, feature_dim, use_temperature, temperature, device, strides=None,
                 mode='raw', use_dist=False):
        super(DCC, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.device = device
        self.mode = mode
        self.use_dist = use_dist

        if mode == 'raw':
            self.encoder = R1DNet(input_channels, mid_channel=16, feature_dim=feature_dim, stride=2,
                                  kernel_size=[7, 11, 11, 7],
                                  final_fc=True)
        elif mode == 'sst':
            # self.encoder = ResNet2d3d(input_size=input_size, input_channel=input_channels, feature_dim=feature_dim)
            self.encoder = Encoder3d(input_size=input_size, input_channel=input_channels, feature_dim=feature_dim)
        elif mode == 'img':
            self.encoder = ResNet(input_channels=input_channels, num_classes=feature_dim)
        else:
            raise ValueError

        self.targets = None

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        if self.mode == 'raw' or self.mode == 'img':
            batch_size, num_epoch, channel, *_ = x.shape
            x = x.view(batch_size * num_epoch, *x.shape[2:])
        else:
            batch_size, num_epoch, time_len, width, height = x.shape
            x = x.view(batch_size * num_epoch, 1, *x.shape[2:])
        feature = self.encoder(x)
        feature = F.normalize(feature, p=2, dim=1)
        feature = feature.view(batch_size, num_epoch, self.feature_dim)

        #################################################################
        #                       Multi-InfoNCE Loss                      #
        #################################################################
        mask = torch.zeros(batch_size, num_epoch, num_epoch, batch_size, dtype=bool)
        for i in range(batch_size):
            for j in range(num_epoch):
                mask[i, j, :, i] = 1
        mask = mask.cuda(self.device)

        logits = torch.einsum('ijk,mnk->ijnm', [feature, feature])
        if self.use_temperature:
            logits /= self.temperature

        pos = torch.exp(logits.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg = torch.exp(logits.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                           batch_size * num_epoch - num_epoch)).sum(-1)

        loss = (-torch.log(pos / (pos + neg))).mean()

        return loss

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class DCCClassifier(nn.Module):
    def __init__(self, input_size, input_channels, feature_dim, num_class, use_l2_norm, use_dropout, use_batch_norm,
                 device, strides=None, mode='raw'):
        super(DCCClassifier, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.device = device
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.mode = mode

        if mode == 'raw':
            self.encoder = Encoder(input_size, input_channels, feature_dim)
        elif mode == 'sst':
            self.encoder = Encoder3d(input_size=input_size, input_channel=input_channels, feature_dim=feature_dim)
        else:
            raise ValueError

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(feature_dim))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_dim, num_class))
        self.final_fc = nn.Sequential(*final_fc)

        # self._initialize_weights(self.final_fc)

    def forward(self, x):
        if self.mode == 'raw':
            batch_size, num_epoch, channel, time_len = x.shape
            x = x.view(batch_size * num_epoch, *x.shape[2:])
        else:
            batch_size, num_epoch, time_len, width, height = x.shape
            x = x.view(batch_size * num_epoch, 1, *x.shape[2:])
        feature = self.encoder(x)
        # feature = feature.view(batch_size, num_epoch, self.feature_dim)

        if self.use_l2_norm:
            feature = F.normalize(feature, p=2, dim=1)

        out = self.final_fc(feature)
        out = out.view(batch_size, num_epoch, -1)

        # print('3. Out: ', out.shape)

        return out


class SSTDIS(nn.Module):
    def __init__(self, input_size_v1, input_size_v2, input_channels, feature_dim, use_temperature, temperature, device,
                 strides=None, first_view='raw'):
        super(SSTDIS, self).__init__()

        assert first_view in ['raw', 'freq']

        self.input_size_v1 = input_size_v1
        self.input_size_v2 = input_size_v2
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.first_view = first_view
        self.device = device
        self.mask = None

        if first_view == 'raw':
            self.encoder_q = Encoder3d(input_size=input_size_v1, input_channel=input_channels,
                                       feature_dim=feature_dim, feature_mode='raw')
            self.encoder_s = Encoder3d(input_size=input_size_v2, input_channel=input_channels,
                                       feature_dim=feature_dim, feature_mode='freq')
        else:
            self.encoder_q = Encoder3d(input_size=input_size_v1, input_channel=input_channels,
                                       feature_dim=feature_dim, feature_mode='freq')
            self.encoder_s = Encoder3d(input_size=input_size_v2, input_channel=input_channels,
                                       feature_dim=feature_dim, feature_mode='raw')

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     '''
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     '''
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()
    #
    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)
    #
    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)
    #
    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this], idx_unshuffle
    #
    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     '''
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     '''
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this]

    def forward(self, x1, x2):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, time_len, width, height = x1.shape
        x1 = x1.view(batch_size * num_epoch, 1, *x1.shape[2:])
        feature_k = self.encoder_q(x1)
        feature_k = F.normalize(feature_k, p=2, dim=1)
        feature_k = feature_k.view(batch_size, num_epoch, self.feature_dim)

        x2 = x2.view(batch_size * num_epoch, 1, *x2.shape[2:])
        feature_s = self.encoder_s(x2)
        feature_s = F.normalize(feature_s, p=2, dim=1)
        feature_s = feature_s.view(batch_size, num_epoch, self.feature_dim)

        #################################################################
        #                       Multi-InfoNCE Loss                      #
        #################################################################
        if self.mask is None:
            mask = torch.zeros(batch_size * 2, num_epoch, num_epoch, batch_size * 2, dtype=bool)
            for i in range(batch_size * 2):
                for j in range(num_epoch):
                    mask[i, j, :, i] = 1
                    mask[i, j, j, (batch_size + i) % (2 * batch_size)] = 1
            mask = mask.cuda(self.device)
            self.mask = mask
        else:
            mask = self.mask

        feature = torch.cat([feature_k, feature_s], dim=0)
        logits = torch.einsum('ijk,mnk->ijnm', [feature, feature])

        pos = torch.exp(logits.masked_select(mask).view(2 * batch_size, num_epoch, num_epoch + 1)).sum(-1)
        neg = torch.exp(logits.masked_select(torch.logical_not(mask)).view(2 * batch_size, num_epoch,
                                                                           2 * batch_size * num_epoch - num_epoch - 1)).sum(
            -1)

        loss = (-torch.log(pos / (pos + neg))).mean()

        return loss

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class SSTMMD(nn.Module):
    def __init__(self, input_size_v1, input_size_v2, input_channels, feature_dim, use_temperature, temperature, device,
                 strides=None, first_view='raw'):
        super(SSTMMD, self).__init__()

        assert first_view in ['raw', 'freq']

        self.input_size_v1 = input_size_v1
        self.input_size_v2 = input_size_v2
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.first_view = first_view
        self.device = device

        if first_view == 'raw':
            self.encoder = Encoder3d(input_size=input_size_v1, input_channel=input_channels,
                                     feature_dim=feature_dim, feature_mode='raw')
            self.sampler = Encoder3d(input_size=input_size_v2, input_channel=input_channels,
                                     feature_dim=feature_dim, feature_mode='freq')
        else:
            self.encoder = Encoder3d(input_size=input_size_v1, input_channel=input_channels,
                                     feature_dim=feature_dim, feature_mode='freq')
            self.sampler = Encoder3d(input_size=input_size_v2, input_channel=input_channels,
                                     feature_dim=feature_dim, feature_mode='raw')

        for param_s in self.sampler.parameters():
            param_s.requires_grad = False  # not update by gradient

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     '''
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     '''
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()
    #
    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)
    #
    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)
    #
    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this], idx_unshuffle
    #
    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     '''
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     '''
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this]

    def forward(self, x1, x2):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, time_len, width, height = x1.shape
        x1 = x1.view(batch_size * num_epoch, 1, *x1.shape[2:])
        feature_k = self.encoder(x1)
        feature_k = F.normalize(feature_k, p=2, dim=1)
        feature_k = feature_k.view(batch_size, num_epoch, self.feature_dim)

        with torch.no_grad():
            x2 = x2.view(batch_size * num_epoch, 1, *x2.shape[2:])
            feature_q = self.sampler(x2)
            feature_q = F.normalize(feature_q, p=2, dim=1)
            feature_q = feature_q.view(batch_size, num_epoch, self.feature_dim)

        #################################################################
        #                       Multi-InfoNCE Loss                      #
        #################################################################
        mask = torch.zeros(batch_size, num_epoch, num_epoch, batch_size, dtype=bool)
        for i in range(batch_size):
            for j in range(num_epoch):
                mask[i, j, :, i] = 1
        mask = mask.cuda(self.device)

        logits = torch.einsum('ijk,mnk->ijnm', [feature_k, feature_k])
        # if self.use_temperature:
        #     logits /= self.temperature

        sim = torch.einsum('ijk,mnk->ijnm', [feature_q, feature_q])

        pos = torch.exp(logits.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg = torch.exp(logits.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                           batch_size * num_epoch - num_epoch))
        neg_v2 = torch.exp(sim.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                           batch_size * num_epoch - num_epoch))
        expand_factor = neg_v2.shape[-1] / neg_v2.sum(-1).unsqueeze(-1)
        neg_v2 *= expand_factor

        # print(pos.max().item(), pos.min().item(), torch.isnan(pos).any().item())
        # print(neg.max().item(), neg.min().item(), torch.isnan(neg).any().item())
        # print(neg_v2.max().item(), neg_v2.min().item(), torch.isnan(neg_v2).any().item())

        neg = (neg * neg_v2).sum(-1)

        loss = (-torch.log(pos / (pos + neg))).mean()

        return loss

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class SSTClassifier(nn.Module):
    def __init__(self, input_size_v1, input_size_v2, input_channels, feature_dim, num_class, use_l2_norm, use_dropout,
                 use_batch_norm,
                 device, strides=None, first_view='raw'):
        super(SSTClassifier, self).__init__()

        self.input_size_v1 = input_size_v1
        self.input_size_v2 = input_size_v2
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.device = device
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm

        if first_view == 'raw':
            self.encoder = Encoder3d(input_size=input_size_v1, input_channel=input_channels, feature_dim=feature_dim,
                                     feature_mode='raw')
            self.sampler = Encoder3d(input_size=input_size_v2, input_channel=input_channels, feature_dim=feature_dim,
                                     feature_mode='freq')
        else:
            self.encoder = Encoder3d(input_size=input_size_v1, input_channel=input_channels, feature_dim=feature_dim,
                                     feature_mode='freq')
            self.sampler = Encoder3d(input_size=input_size_v2, input_channel=input_channels, feature_dim=feature_dim,
                                     feature_mode='raw')

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(feature_dim * 2))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_dim * 2, num_class))
        self.final_fc = nn.Sequential(*final_fc)

    def forward(self, x1, x2):
        batch_size, num_epoch, *_ = x1.shape

        x1 = x1.view(x1.shape[0] * x1.shape[1], 1, *x1.shape[2:])
        x2 = x2.view(x2.shape[0] * x2.shape[1], 1, *x2.shape[2:])

        feature1 = self.encoder(x1)
        feature2 = self.sampler(x2)

        if self.use_l2_norm:
            feature1 = F.normalize(feature1, p=2, dim=1)
            feature2 = F.normalize(feature2, p=2, dim=1)

        feature = torch.cat([feature1, feature2], dim=-1)

        out = self.final_fc(feature)
        out = out.view(batch_size, num_epoch, -1)

        return out


if __name__ == '__main__':
    model = DCC(input_size=200, input_channels=62, feature_dim=128, use_temperature=True, temperature=0.07,
                device=0, strides=None, mode='raw')
    model = model.cuda()
    out = model(torch.randn(16, 10, 62, 200).cuda())
    print(out)
