'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2023-10-30
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


class CriteoBaseDataset(Dataset):

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


class CriteoProcessData():

    def __init__(self, config: dict) -> None:
        self.config = self.process_config(config)
        self.df = pd.read_csv(get_file_path(self.config["data_path"]))
        logger.info("配置读取成功，数据读取成功")

    @staticmethod
    @config_middleware()
    def process_config(config):
        return config

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # 数据集处理
        if self.config["debug_mode"]:
            self.df = self.df[:10000]
            logger.info("当前模式为debug模式，读取前10000条数据")
        else:
            logger.info("当前模式非debug模式，读取全量数据")
        self.df[self.config["dense_cols"]] = self.df[self.config["dense_cols"]].fillna(0)
        self.df[self.config["sparse_cols"]] = self.df[self.config["sparse_cols"]].fillna('-1')

        # 切分数据集
        train_df = self.df[:int(len(self.df) * self.config["train_ratio"])].reset_index(drop=True)
        valid_df = self.df[int(len(self.df) * self.config["train_ratio"]):int(len(self.df) * (self.config["train_ratio"] + self.config["valid_ratio"]))].reset_index(drop=True)
        test_df = self.df[int(len(self.df) * (self.config["train_ratio"] + self.config["valid_ratio"])):].reset_index(drop=True)
        logger.info("数据集切分完成，其中训练集、验证集、测试集的比例为：{:.1f}:{:.1f}:{:.1f}".format(self.config["train_ratio"], self.config["valid_ratio"], 1 - self.config["train_ratio"] - self.config["valid_ratio"]))
        return train_df, valid_df, test_df

    def get_dataset(self, df: pd.DataFrame, enc_dict: dict = None) -> CriteoBaseDataset:
        dataset = CriteoBaseDataset(df, self.config, enc_dict)
        return dataset

    def get_dataloader(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def data_process(self):
        train_df, valid_df, test_df = self.split_data()
        train_dataset = self.get_dataset(train_df)
        valid_dataset = self.get_dataset(valid_df, train_dataset.enc_dict)
        test_dataset = self.get_dataset(test_df, train_dataset.enc_dict)
        train_dataloader = self.get_dataloader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        valid_dataloader = self.get_dataloader(valid_dataset, batch_size=self.config["batch_size"], shuffle=False)
        test_dataloader = self.get_dataloader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)
        logger.info("dataloader处理完成，其中训练集、验证集、测试集的batch_size为：{}:{}:{}".format(self.config["batch_size"], self.config["batch_size"], self.config["batch_size"]))
        return train_dataloader, valid_dataloader, test_dataloader, train_dataset.enc_dict
