'''
Author       : wyx-hhhh
Date         : 2024-04-08
LastEditTime : 2024-07-18
Description  : 
'''
import os
import pandas as pd
from torch.utils.data import Dataset
from data.gowalla_dgl.dataset import get_gowalla_dgl_dataset
from managers.logger_manager import logger
from data.base_process_data import BaseProcessData
from utils.file_utils import get_file_path


class GowallaProcessDglData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.config = config

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # 判断数据集是否处理过
        data_path = get_file_path(["data", "gowalla", "loc-gowalla_totalCheckins.txt"])
        data = pd.read_csv(data_path, sep='\t', header=None, names=["user_id", "check_in_time", "latitude", "longitude", "item_id"])
        if os.path.exists(get_file_path(self.config["data_path"])):
            logger.info("数据集已处理过，直接读取")
            train_data = pd.read_csv(get_file_path(self.config["train_path"]))
            test_data = pd.read_csv(get_file_path(self.config["test_path"]))
            summary_data = pd.concat([train_data, test_data])
            self.config["user_num"] = summary_data["user_id"].nunique()
            self.config["item_num"] = summary_data["item_id"].nunique()
        else:
            logger.info("数据集未处理过，开始处理数据集")
            data = data[["user_id", "item_id"]]
            # 去除重复数据，去除空数据
            data = data.drop_duplicates()
            data = data.dropna()
            # 去除交互次数小于10的用户
            user_count = data["user_id"].value_counts()
            user_count = user_count[user_count >= 11]
            data = data[data["user_id"].isin(user_count.index)]
            # 重新映射user_id和item_id
            user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_count.index)}
            item_id_map = {old_id: new_id for new_id, old_id in enumerate(data["item_id"].unique())}
            data["old_user_id"] = data["user_id"]
            data["old_item_id"] = data["item_id"]
            data["user_id"] = data["user_id"].map(user_id_map).astype(int)
            data["item_id"] = data["item_id"].map(item_id_map).astype(int)
            data.to_csv(get_file_path(["data", "gowalla", "gowalla.csv"]), index=False)
            # 数据集处理，分割成训练集、测试集，比例为8:2，并进行存储
            train_data = data.sample(frac=self.config.get("train_ratio", 0.8), random_state=2024)[["user_id", "item_id"]]
            test_data = data.drop(train_data.index)[["user_id", "item_id"]]
            train_data.to_csv(get_file_path(["data", "gowalla", "train.csv"]), index=False)
            test_data.to_csv(get_file_path(["data", "gowalla", "test.csv"]), index=False)
            self.config["user_num"] = len(user_id_map)
            self.config["item_num"] = len(item_id_map)
        if self.config["debug_mode"]:
            train_data = train_data[:10000]
            test_data = test_data[:10000]
            logger.info("当前模式为debug模式，读取前10000条数据")
        return train_data, pd.DataFrame(), test_data

    def data_process(self):
        result_dict = {}
        train_df, valid_df, test_df = self.split_data()
        train_dataset = self.get_dataset(train_df, "train")
        valid_dataset = None if valid_df.empty else self.get_dataset(valid_df, "valid")
        test_dataset = self.get_dataset(test_df, "test")
        graph_data = train_dataset.generate_graph()
        train_dataloader = self.get_dataloader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        valid_dataloader = None if valid_df.empty else self.get_dataloader(valid_dataset, batch_size=self.config["batch_size"], shuffle=False)
        test_dataloader = self.get_dataloader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)
        result_dict['train_df'] = train_df
        result_dict['valid_df'] = valid_df
        result_dict['test_df'] = test_df
        result_dict["train_grouped_data"] = train_dataset.grouped_data
        result_dict["valid_grouped_data"] = valid_dataset.grouped_data if valid_dataset else None
        result_dict["test_grouped_data"] = test_dataset.grouped_data
        result_dict["train_dataloader"] = train_dataloader
        result_dict["valid_dataloader"] = valid_dataloader
        result_dict["test_dataloader"] = test_dataloader
        result_dict["graph_data"] = graph_data
        logger.info("dataloader处理完成，其中训练集、验证集、测试集的batch_size为：{}:{}:{}".format(self.config["batch_size"], self.config["batch_size"], self.config["batch_size"]))
        logger.send_message(message="数据处理处理完成", message_type=2, message_content_type=1)
        return result_dict

    def get_dataset(self, data: pd.DataFrame, phase: str) -> Dataset:
        dataset = get_gowalla_dgl_dataset(data, self.config, phase)
        return dataset
