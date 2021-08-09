"""
@Time    : 2021/8/10 1:15
@File    : test_datasets.py
@Software: PyCharm
@Desc    : 
"""
import os

from mme.data.datasets import ISRUCDataset


# def test_deap():
#     data_path = '/data/DataHub/EmotionRecognition/DEAP/signal'
#     dataset = DEAPDataset(data_path, num_seq=10, subject_list=list(range(5)), modal='eeg')
#     print(dataset[0][0].shape, dataset.channels)

# def test_sleepedf():
#     data_path = '/data/DataHub/SleepClassification/sleepedf153/sleepedf153'
#     files = os.listdir(data_path)
#     dataset = SleepEDFDataset(data_path=data_path, num_epoch=10,
#                               transform=None, patients=files[:10], preprocessing='none', modal='eeg')
#     print(dataset[0][0].shape, dataset[0][1].shape, dataset.channels)


def test_isruc():
    data_path = '/data/DataHub/SleepClassification/isruc/isruc_mat/subgroup1'
    files = os.listdir(data_path)
    dataset = ISRUCDataset(data_path=data_path, num_epoch=10,
                           transform=None, patients=files[:10], preprocessing='none', modal='emg')
    print(dataset[0][0].shape, dataset[0][1].shape, dataset.channels)
