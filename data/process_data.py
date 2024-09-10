'''
Author       : wyx-hhhh
Date         : 2024-06-12
LastEditTime : 2024-09-04
Description  : 
'''
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data.dataset import get_dataset
from utils.utilities import get_file_path


class ProcessData:

    def __init__(self, config: dict) -> None:
        self.config = config

    def get_dataset(self, data: pd.DataFrame, phase: str) -> Dataset:
        dataset = get_dataset(data, self.config, phase)
        return dataset

    def get_dataloader(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False) -> DataLoader:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data_path = get_file_path(self.config["train_path"])
        test_data_path = get_file_path(self.config["test_path"])

        train_df = pd.read_table(train_data_path, sep=' ', names=['user_id', 'item_id'])
        test_df = pd.read_table(test_data_path, sep=' ', names=['user_id', 'item_id'])
        user_num = max(train_df['user_id'].max(), test_df['user_id'].max()) + 1
        item_num = max(train_df['item_id'].max(), test_df['item_id'].max()) + 1

        process_data_config = {
            "user_num": user_num,
            "item_num": item_num,
            "train_datasize": len(train_df),
            "test_datasize": len(test_df),
            "users_average_actions": len(train_df) / (user_num),
            "items_average_actions": len(train_df) / (item_num),
            "inters_num": len(train_df) + len(test_df),
            "sparsity": (len(train_df) + len(test_df)) / ((user_num) * (item_num)),
        }
        self.config["logger"].info(f"Process Data Info:")
        for key, value in process_data_config.items():
            self.config["logger"].info(f"{key}: {value}")
        self.config["logger"].info("\n")
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
        self.config["logger"].send_message(message="数据处理处理完成", message_type=2, message_content_type=1)
        return result_dict
