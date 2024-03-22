import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import torch
from typing import List
from tqdm import tqdm

from trainers.base_train import BaseTrainer


class CriteoTrainer(BaseTrainer):

    def __init__(self):
        super(CriteoTrainer, self).__init__()
