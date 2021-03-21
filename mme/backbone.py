"""
@Time    : 2021/3/20 22:10
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : backbone.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, input_channel, feature_dim):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            # nn.Dropout(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            # nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        last_size = input_size
        for i in range(3):
            last_size //= 2

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * last_size, feature_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(feature_dim, eps=0.001),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = Encoder(62, 16)
    x = torch.randn(32, 10, 62, 200)  # (batch, seq_len, channels, time_len)
    x = x.view(-1, *x.shape[2:])
    out = model(x)
