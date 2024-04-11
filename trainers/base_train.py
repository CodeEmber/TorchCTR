'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-04-11
Description  : 
'''
from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm


class BaseTrainer():

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def test_model(self, *args, **kwargs):
        pass

    def valid_model(self, *args, **kwargs):
        raise NotImplementedError()

    def get_test_predict(self, *args, **kwargs):
        raise NotImplementedError()
