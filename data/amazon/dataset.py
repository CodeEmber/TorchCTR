'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-19
Description  : 
'''
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import numpy as np

from utils.file_utils import get_file_path
from utils.middleware import config_middleware
from utils.logger import MyLogger

logger = MyLogger()


class AmazonBaseDataset(Dataset):

    def __init__(self, df: pd.DataFrame, config: dict, enc_dict: dict = None):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        self.sparse_cols = list(set(self.config["sparse_cols"]))
        self.history_cols = list(set(self.config["history_cols"]))
        self.features_name = self.sparse_cols + self.history_cols + ["label"]

        if self.enc_dict is None:
            self.enc_dict = self.get_enc_dict()
        self.enc_data()

    def get_enc_dict(self) -> dict:
        self.enc_dict = dict(zip(list(self.sparse_cols + self.history_cols), [dict() for _ in range(len(self.sparse_cols + self.history_cols))]))
        self.enc_dict["user_id"] = {'vocab_size': self.df["user_id"].nunique() + 1, 'type': 'user'}
        self.enc_dict["item_target_id"] = {'vocab_size': self.df["item_target_id"].nunique() + 1, 'type': 'item'}
        self.enc_dict["item_target_category"] = {'vocab_size': self.df["item_target_category"].nunique() + 1, 'type': 'item'}
        self.enc_dict["item_history_id"] = {'share_with': 'item_target_id', 'type': 'item'}
        self.enc_dict["item_history_category"] = {'share_with': 'item_target_category', 'type': 'item'}
        return self.enc_dict

    def enc_data(self):
        self.enc_data_dict = defaultdict(lambda: np.array([]))

        for col in self.sparse_cols:
            self.enc_data_dict[col] = torch.Tensor(np.array(self.df[col].values)).long()

        for col in self.history_cols:
            self.enc_data_dict[col] = torch.Tensor(np.array(self.df[col].values.tolist())).long()

    def __getitem__(self, index) -> dict:
        data = defaultdict(lambda: np.array([]))
        for col in self.features_name:
            if col in self.sparse_cols + self.history_cols:
                data[col] = self.enc_data_dict[col][index]
            else:
                data[col] = torch.Tensor([self.df['label'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.df)


def get_amazon_dataset(data: str, config: dict, enc_dict: dict) -> AmazonBaseDataset:
    dataset = AmazonBaseDataset(data, config, enc_dict)
    return dataset
