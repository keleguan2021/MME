#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Xiao Qinfeng
# @Date:   2021/8/5 10:43
# @Last Modified by:   Xiao Qinfeng
# @Last Modified time: 2021/8/5 10:43
# @Software: PyCharm

from visbrain.io import ReadSleepData

FILE_NAME = 'data/isruc/subgroup1/1/1.rec'


if __name__ == '__main__':
    data = ReadSleepData(data=FILE_NAME)
