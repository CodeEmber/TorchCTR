'''
Author       : wyx-hhhh
Date         : 2024-05-28
LastEditTime : 2024-08-14
Description  : 
'''
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from managers.logger_manager import logger


class BaseProcessData():

    def __init__(self, config: dict) -> None:
        self.config = config

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def get_dataset(self, data: pd.DataFrame, enc_dict: dict = None) -> Dataset:
        raise NotImplementedError

    def get_dataloader(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def data_process(self):
        result_dict = {}
        train_df, valid_df, test_df = self.split_data()
        train_dataset = self.get_dataset(train_df)
        valid_dataset = None if valid_df.empty else self.get_dataset(valid_df, train_dataset.enc_dict)
        test_dataset = self.get_dataset(test_df, train_dataset.enc_dict)
        train_dataloader = self.get_dataloader(train_dataset, batch_size=self.config["batch_size"], shuffle=True, drop_last=self.config.get("drop_last", False))
        valid_dataloader = None if valid_df.empty else self.get_dataloader(valid_dataset, batch_size=self.config["batch_size"], shuffle=False, drop_last=self.config.get("drop_last", False))
        test_dataloader = self.get_dataloader(test_dataset, batch_size=self.config["batch_size"], shuffle=False, drop_last=self.config.get("drop_last", False))
        result_dict['train_df'] = train_df
        result_dict['valid_df'] = valid_df
        result_dict['test_df'] = test_df
        result_dict["train_dataloader"] = train_dataloader
        result_dict["valid_dataloader"] = valid_dataloader
        result_dict["test_dataloader"] = test_dataloader
        result_dict["enc_dict"] = train_dataset.enc_dict
        logger.send_message(message="数据处理处理完成", message_type=2, message_content_type=1)
        return result_dict
