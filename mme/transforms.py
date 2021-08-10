#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Xiao Qinfeng
# @Date:   2021/8/5 10:30
# @Last Modified by:   Xiao Qinfeng
# @Last Modified time: 2021/8/5 10:30
# @Software: PyCharm

import warnings
from typing import Union, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from .functional import spectrogram


class Spectrogram(object):
    def __init__(self, fs: float, window: str='hann', nperseg: int=400, nfft: int=400):
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.nfft = nfft

    def __call__(self, x: np.ndarray):
        return spectrogram(x, fs=self.fs, window=self.window, nperseg=self.nperseg, nfft=self.nfft)
