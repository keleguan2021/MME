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
from .backbone_2d import ResNet
from .backbone_3d import ResNet2d3d, Encoder3d

if __name__ == '__main__':
    model = ResNet2d3d(input_channel=5, feature_dim=128)
    x = torch.randn(32 * 10, 5, 200, 32, 32)  # (batch, seq_len, channels, time_len)
    out = model(x)
    print(out.shape)
