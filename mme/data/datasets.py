"""
@Time    : 2021/2/6 15:20
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""

# from .utils import tensor_standardize
from .amigos import AMIGOSDataset
from .deap import DEAPDataset
from .isruc import ISRUCDataset
from .seed import SEEDDataset
from .seediv import SEEDIVDataset
from .sleepedf import SleepEDFDataset

contents = [AMIGOSDataset, DEAPDataset, ISRUCDataset, SEEDIVDataset, SEEDIVDataset, SleepEDFDataset]
