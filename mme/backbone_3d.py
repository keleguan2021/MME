"""
@Time    : 2021/3/30 22:19
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : backbone_3d.py
@Software: PyCharm
@Desc    : 
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias)


def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    # 1x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=bias)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = torch.cat([out, zero_pads], dim=1)

    return out


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(BasicBlock3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(BasicBlock2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv1x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(Bottleneck3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class Bottleneck2d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(Bottleneck2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1),
                               bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm: out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class ResNet2d3d(nn.Module):
    def __init__(self, input_channel, input_size, feature_dim, strides=None, block=None, layers=None, use_final_fc=True,
                 track_running_stats=True):
        super(ResNet2d3d, self).__init__()

        self.input_channel = input_channel
        self.input_size = list(input_size)
        self.feature_dim = feature_dim
        self.use_final_fc = use_final_fc

        if strides is None:
            strides = [1, 1, 2, 2]
        if layers is None:
            layers = [2, 2, 2, 2]
        if block is None:
            block = [BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d]

        self.inplanes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=bias)
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        if not isinstance(block, list):
            block = [block] * 4

        self.layer1 = self._make_layer(block[0], 64, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block[3], 256, layers[3], stride=strides[3], is_final=True)

        self.input_size[1] = int(math.ceil(self.input_size[1] / 4))
        self.input_size[2] = int(math.ceil(self.input_size[2] / 4))
        for idx, block_item in enumerate(block):
            if block_item == BasicBlock2d:
                self.input_size[1] = int(math.ceil(self.input_size[1] / strides[idx]))
                self.input_size[2] = int(math.ceil(self.input_size[2] / strides[idx]))
            elif block_item == BasicBlock3d:
                self.input_size[0] = int(math.ceil(self.input_size[0] / strides[idx]))
                self.input_size[1] = int(math.ceil(self.input_size[1] / strides[idx]))
                self.input_size[2] = int(math.ceil(self.input_size[2] / strides[idx]))
            else:
                raise ValueError

        if use_final_fc:
            self.final_fc = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self.input_size[0] * self.input_size[1] * self.input_size[2] * 256, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )

        # modify layer4 from exp=512 to exp=256
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=customized_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion, track_running_stats=self.track_running_stats)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, track_running_stats=self.track_running_stats))
        self.inplanes = planes * block.expansion
        if is_final:  # if is final block, no ReLU in the final output
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats))
            layers.append(
                block(self.inplanes, planes, track_running_stats=self.track_running_stats, use_final_relu=False))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_final_fc:
            x = x.view(x.size(0), -1)
            x = self.final_fc(x)

        return x


class Encoder3d(nn.Module):
    def __init__(self, input_size, input_channel, feature_dim, feature_mode='raw'):
        super(Encoder3d, self).__init__()

        if feature_mode == 'raw':
            strides = (2, 2, 2, 2)
        elif feature_mode == 'freq':
            strides = (1, 1, 1, 1)
        else:
            raise ValueError

        self.features = nn.Sequential(
            # Heading conv layer
            nn.Conv3d(input_channel, 64, kernel_size=(3, 3, 3), stride=(strides[0], 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(strides[1], 2, 2), padding=(1, 1, 1)),

            # First group of conv layer (2d, without stride)
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Second group of conv layer with stride (3d)
            nn.Conv3d(128, 128, kernel_size=3, stride=(strides[2], 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # Third group of conv layer with stride (3d)
            nn.Conv3d(256, 256, kernel_size=3, stride=(strides[3], 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )

        last_size = list(input_size)
        last_size[0] = int(math.ceil(last_size[0] / strides[0]))
        last_size[1] = int(math.ceil((last_size[1] / 2)))
        last_size[2] = int(math.ceil((last_size[2] / 2)))
        last_size[0] = int(math.ceil(last_size[0] / strides[1]))
        last_size[1] = int(math.ceil((last_size[1] / 2)))
        last_size[2] = int(math.ceil((last_size[2] / 2)))

        last_size[0] = int(math.ceil(last_size[0] / strides[2]))
        last_size[1] = int(math.ceil((last_size[1] / 2)))
        last_size[2] = int(math.ceil((last_size[2] / 2)))

        last_size[0] = int(math.ceil(last_size[0] / strides[3]))
        last_size[1] = int(math.ceil((last_size[1] / 2)))
        last_size[2] = int(math.ceil((last_size[2] / 2)))

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * last_size[0] * last_size[1] * last_size[2], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    model = Encoder3d(input_channel=1, input_size=(5, 32, 32), feature_dim=128, feature_mode='freq')
    out = model(torch.randn(32, 1, 5, 32, 32))
    print(out.shape)
