'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-02-27
Description  : 
'''
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data.criteo.dataset import get_criteo_dataset
from data.movie_len.dataset import MovieLenBaseDataset, get_movie_len_dataset

from utils.file_utils import get_file_path
from utils.middleware import config_middleware
from utils.logger import MyLogger

logger = MyLogger()


class BaseProcessData():

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

    def get_dataset(self, data: pd.DataFrame, enc_dict: dict = None) -> Dataset:
        if self.config["data"] == "criteo":
            dataset = get_criteo_dataset(data, self.config, enc_dict)
        elif self.config["data"] == "movielens":
            dataset = get_movie_len_dataset(data, self.config, enc_dict)
        else:
            raise ValueError("数据集路径错误")
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


class CriteoProcessData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    @staticmethod
    @config_middleware()
    def process_config(config):
        return config


class MovieLenProcessData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    @staticmethod
    @config_middleware()
    def process_config(config):
        return config


class ProcessData():

    def __init__(self, config: dict) -> None:
        self.config = config
        self.data = self.get_data()

    def get_data(self):
        if self.config["data"] == "criteo":
            data = CriteoProcessData(config=self.config)
        elif self.config["data"] == "movielens":
            data = MovieLenProcessData(config=self.config)
        else:
            raise ValueError("数据集错误")
        return data

    def data_process(self):
        return self.data.data_process()
