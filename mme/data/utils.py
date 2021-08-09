#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Xiao Qinfeng
# @Date:   2021/8/9 11:26
# @Last Modified by:   Xiao Qinfeng
# @Last Modified time: 2021/8/9 11:26
# @Software: PyCharm

import numpy as np

EPS = 1e-8


def tackle_denominator(x: np.ndarray):
    x[x == 0.0] = EPS
    return x


def tensor_standardize(x: np.ndarray, dim=-1):
    x_mean = np.expand_dims(x.mean(axis=dim), axis=dim)
    x_std = np.expand_dims(x.std(axis=dim), axis=dim)
    return (x - x_mean) / tackle_denominator(x_std)
