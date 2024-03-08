'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-02-27
Description  : 
'''
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import numpy as np

from utils.file_utils import get_file_path
from utils.middleware import config_middleware
from utils.logger import MyLogger

logger = MyLogger()


class MovieLenBaseDataset(Dataset):

    def __init__(self, df: pd.DataFrame, config: dict, enc_dict: dict = None):
        self.config = config
        self.df = df
        self.enc_df = pd.DataFrame()
        self.enc_dict = enc_dict
        self.dense_cols = list(set(self.config["dense_cols"]))
        self.sparse_cols = list(set(self.config["sparse_cols"]))
        self.features_name = self.dense_cols + self.sparse_cols + ["label"]

        if self.enc_dict is None:
            self.enc_dict = self.get_enc_dict()
        self.enc_data()

    def get_enc_dict(self) -> dict:
        self.enc_dict = dict(zip(list(self.dense_cols + self.sparse_cols), [dict() for _ in range(len(self.dense_cols + self.sparse_cols))]))
        for col in self.sparse_cols:
            self.df.loc[:, col] = self.df[col].astype(str)
            us = self.df[col].unique()
            self.enc_dict[col] = dict(zip(us, range(1, len(us) + 1)))
            self.enc_dict[col]["vocab_size"] = len(us) + 1

        for col in self.dense_cols:
            self.df.loc[:, col] = self.df[col].astype(float)
            self.enc_dict[col]["max"] = self.df.loc[:, col].max()
            self.enc_dict[col]["min"] = self.df.loc[:, col].min()

        return self.enc_dict

    def enc_dense_data(self, col: str) -> int:
        return (self.df.loc[:, col] - self.enc_dict[col]["min"]) / (self.enc_dict[col]["max"] - self.enc_dict[col]["min"])

    def enc_sparse_data(self, col: str) -> np.ndarray:
        return self.df.loc[:, col].map(self.enc_dict[col]).fillna(0).astype(int)

    def enc_data(self):
        self.enc_df = copy.deepcopy(self.df)
        for col in self.dense_cols:
            self.enc_df.loc[:, col] = self.enc_dense_data(col)
        for col in self.sparse_cols:
            self.enc_df.loc[:, col] = self.enc_sparse_data(col)

    def __getitem__(self, index) -> dict:
        data = dict()
        for col in self.features_name:
            if col in self.sparse_cols:
                data[col] = torch.tensor(self.enc_df[col][index], dtype=torch.long).squeeze(-1)
            elif col in self.dense_cols:
                data[col] = torch.tensor(self.enc_df[col][index], dtype=torch.float32).squeeze(-1)
            else:
                data[col] = torch.tensor(self.enc_df[col][index]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.enc_df)


def get_movie_len_dataset(data: str, config: dict, enc_dict: dict) -> MovieLenBaseDataset:
    dataset = MovieLenBaseDataset(data, config, enc_dict)
    return dataset
