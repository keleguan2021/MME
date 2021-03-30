"""
@Time    : 2021/3/20 22:10
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : backbone.py
@Software: PyCharm
@Desc    : 
"""
import torch

from .backbone_1d import Encoder, R1DNet
from .backbone_resnet3d import ResNet2d3d

if __name__ == '__main__':
    model = Encoder(62, 16)
    x = torch.randn(32, 10, 62, 200)  # (batch, seq_len, channels, time_len)
    x = x.view(-1, *x.shape[2:])
    out = model(x)
