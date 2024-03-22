'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-03-22
Description  : 
'''
import pandas as pd
from torch.utils.data import Dataset
from utils.logger import logger
from data.base_process_data import BaseProcessData
from utils.file_utils import get_file_path
from data.criteo.dataset import get_criteo_dataset


class CriteoProcessData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.df = pd.read_csv(get_file_path(config["data_path"]))

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
        dataset = get_criteo_dataset(data, self.config, enc_dict)
        return dataset
