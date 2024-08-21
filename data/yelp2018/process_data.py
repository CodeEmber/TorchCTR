'''
Author       : wyx-hhhh
Date         : 2024-04-08
LastEditTime : 2024-08-14
Description  : 
'''
import pandas as pd
from torch.utils.data import Dataset
from data.gowalla_matrix.dataset import get_gowalla_matrix_dataset
from managers.logger_manager import logger
from data.base_process_data import BaseProcessData
from utils.file_utils import get_file_path
import numpy as np


class YelpProcessData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.config = config

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data_path = get_file_path(self.config["train_path"])
        test_data_path = get_file_path(self.config["test_path"])
        user_list_path = get_file_path(self.config["user_list_path"])
        item_list_path = get_file_path(self.config["item_list_path"])
        train_data = []
        user_num = 0
        item_num = 0
        with open(train_data_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    train_data += [(int(l[0]), i) for i in items]
                    user_num = max(user_num, int(l[0]))
                    item_num = max(item_num, max(items))

        test_data = []
        with open(test_data_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    test_data += [(int(l[0]), i) for i in items]
                    user_num = max(user_num, int(l[0]))
                    item_num = max(item_num, max(items))
        train_df = pd.DataFrame(train_data, columns=["user_id", "item_id"])
        test_df = pd.DataFrame(test_data, columns=["user_id", "item_id"])

        process_data_config = {
            "user_num": user_num + 1,
            "item_num": item_num + 1,
            "train_datasize": len(train_data),
            "test_datasize": len(test_data),
            "users_average_actions": len(train_data) / (user_num + 1),
            "items_average_actions": len(train_data) / (item_num + 1),
            "inters_num": len(train_data) + len(test_data),
            "sparsity": (len(train_data) + len(test_data)) / ((user_num + 1) * (item_num + 1)),
        }
        logger.info(f"Process Data Info:")
        for key, value in process_data_config.items():
            logger.info(f"{key}: {value}")
        logger.info("\n")
        self.config.update(process_data_config)
        return train_df, pd.DataFrame(), test_df

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
        logger.send_message(message="数据处理处理完成", message_type=2, message_content_type=1)
        return result_dict

    def get_dataset(self, data: pd.DataFrame, phase: str) -> Dataset:
        dataset = get_gowalla_matrix_dataset(data, self.config, phase)
        return dataset
