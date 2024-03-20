'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-20
Description  : 
'''
from cProfile import label
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.amazon.dataset import get_amazon_dataset
from data.criteo.dataset import get_criteo_dataset
from data.movielens.dataset import get_movie_len_dataset

from utils.file_utils import get_file_path
from utils.middleware import config_middleware
from utils.logger import MyLogger

logger = MyLogger()


class BaseProcessData():

    def __init__(self, config: dict) -> None:
        self.config = self.process_config(config)
        logger.info("配置读取成功")

    @staticmethod
    @config_middleware()
    def process_config(config):
        return config

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def get_dataset(self, data: pd.DataFrame, enc_dict: dict = None) -> Dataset:
        raise NotImplementedError

    def get_dataloader(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def data_process(self):
        raise NotImplementedError


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


class MovieLenProcessData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # self.user_df = pd.read_csv(
        #     get_file_path(config["user_path"]),
        #     sep='\t',
        #     header=None,
        #     names=["user_id", "age", "gender", "occupation", "zip_code"],
        # )
        # self.item_df = pd.read_csv(
        #     get_file_path(config["item_path"]),
        #     sep='\t',
        #     header=None,
        #     names=["item_id", "movie_title", "release_year", "genre"],
        # )

    def negative_sampling(self, pos_dict, item_list, item_num, ratio, is_test=False, test_num=0):
        neg_samples = []
        desc = "测试集负采样" if is_test else "训练集负采样"
        for user, pos_items in tqdm(pos_dict.items(), desc=desc):
            user_neg_samples = []
            if is_test:
                neg_sample_per_user = test_num
            else:
                user_count = len(pos_items) - 1
                neg_sample_per_user = user_count * ratio
            while len(user_neg_samples) < neg_sample_per_user:
                temp_item_index = random.randint(0, item_num - 1)
                if item_list[temp_item_index] not in pos_items:
                    user_neg_samples.append(item_list[temp_item_index])
            neg_samples.extend([(user, item, 0) for item in user_neg_samples])
        return neg_samples

    def construct_data(self, pos_df, pos_dict, ratio):
        train_user_list = []
        train_item_list = []
        train_label_list = []

        test_user_list = []
        test_item_list = []
        test_label_list = []

        if self.config['debug_mode']:
            user_list = pos_df['user_id'].unique()[:1000]
        else:
            user_list = pos_df['user_id'].unique()

        item_list = pos_df['item_id'].unique()
        item_num = pos_df['item_id'].nunique()

        for user in tqdm(user_list, desc="构建正样本数据"):
            # 训练集正样本
            for i in range(len(pos_dict[user]) - 1):
                train_user_list.append(user)
                train_item_list.append(pos_dict[user][i])
                train_label_list.append(1)

            # 测试集正样本
            test_user_list.append(user)
            test_item_list.append(pos_dict[user][-1])
            test_label_list.append(1)

        # 负采样
        neg_samples_train = self.negative_sampling(pos_dict=pos_dict, item_list=item_list, item_num=item_num, ratio=ratio)
        neg_samples_test = self.negative_sampling(pos_dict=pos_dict, item_list=item_list, item_num=item_num, ratio=ratio, is_test=True, test_num=100)
        train_df = pd.DataFrame(data={
            'user_id': train_user_list + [sample[0] for sample in neg_samples_train],
            'item_id': train_item_list + [sample[1] for sample in neg_samples_train],
            'label': train_label_list + [sample[2] for sample in neg_samples_train],
        })
        test_df = pd.DataFrame(data={
            'user_id': test_user_list + [sample[0] for sample in neg_samples_test],
            'item_id': test_item_list + [sample[1] for sample in neg_samples_test],
            'label': test_label_list + [sample[2] for sample in neg_samples_test],
        })
        return train_df, test_df

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.behaviour_df = pd.read_csv(get_file_path(self.config["behaviour_path"]), sep='\t')
        self.behaviour_df = self.behaviour_df.rename(columns={k: k.split(':')[0] for k in self.behaviour_df.columns})
        self.behaviour_df['user_count'] = self.behaviour_df['user_id'].map(self.behaviour_df['user_id'].value_counts())
        self.behaviour_df = self.behaviour_df[self.behaviour_df['user_count'] > 20].reset_index(drop=True)
        pos_df = self.behaviour_df[self.behaviour_df['rating'] > 3].reset_index(drop=True)
        pos_df = pos_df.sort_values(by=['user_id', 'timestamp'], ascending=True)
        pos_dict = pos_df.groupby('user_id')['item_id'].apply(list).to_dict()
        train_df, test_df = self.construct_data(pos_df, pos_dict, self.config["neg_sample_ratio"])
        return train_df, test_df

    def get_dataset(self, data: pd.DataFrame, enc_dict: dict = None) -> Dataset:
        dataset = get_movie_len_dataset(data, self.config, enc_dict)
        return dataset

    def data_process(self):
        train_df, test_df = self.split_data()
        train_dataset = self.get_dataset(train_df)
        test_dataset = self.get_dataset(test_df, train_dataset.enc_dict)
        train_dataloader = self.get_dataloader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        test_dataloader = self.get_dataloader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)
        valid_dataloader = None
        logger.info("dataloader处理完成，其中训练集、测试集的batch_size为：{}:{}".format(self.config["batch_size"], self.config["batch_size"]))
        return train_dataloader, valid_dataloader, test_dataloader, train_dataset.enc_dict


class AmazonProcessData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def load_data(self):
        self.behaviour_df = pd.read_csv(get_file_path(self.config["behaviour_path"]), sep='\t')
        self.behaviour_df = self.behaviour_df.rename(columns={k: k.split(':')[0] for k in self.behaviour_df.columns})
        self.item_df = pd.read_csv(get_file_path(self.config["item_path"]), sep='\t')
        self.item_df = self.item_df.rename(columns={k: k.split(':')[0] for k in self.item_df.columns})

        # 对数据进行处理
        pos_df = self.behaviour_df[self.behaviour_df['rating'] > 3].reset_index(drop=True)
        pos_df['user_count'] = pos_df['user_id'].map(pos_df['user_id'].value_counts())
        pos_df = pos_df[pos_df['user_count'] > 5].reset_index(drop=True)
        pos_df = pos_df.sort_values(by=['user_id', 'timestamp'], ascending=True)
        # 处理item数据，取最后一个category
        self.item_df['categories'] = self.item_df['categories'].apply(lambda x: eval(x)[-1])
        pos_df = pos_df.merge(self.item_df[['item_id', 'categories']], on='item_id', how='left').reset_index(drop=True)

        # 数据映射，将user_id、item_id、categories映射为从1开始的整数
        item_map_dict = dict(zip(pos_df['item_id'].unique(), range(1, pos_df['item_id'].nunique() + 1)))
        pos_df['item_id'] = pos_df['item_id'].map(item_map_dict)
        user_map_dict = dict(zip(pos_df['user_id'].unique(), range(1, pos_df['user_id'].nunique() + 1)))
        pos_df['user_id'] = pos_df['user_id'].map(user_map_dict)
        categories_map_dict = dict(zip(pos_df['categories'].unique(), range(1, pos_df['categories'].nunique() + 1)))
        pos_df['categories'] = pos_df['categories'].map(categories_map_dict)
        pos_dict = pos_df.groupby('user_id')['item_id'].apply(list).to_dict()
        logger.info("数据加载完成")
        return pos_df, pos_dict

    def construct_data(self, pos_df, pos_dict, max_seq=20, neg_sample_ratio=2):
        label_list = []
        user_id_list = []
        item_target_id_list = []
        item_target_category_list = []
        item_history_seq_id_list = []
        item_history_seq_category_list = []
        if self.config['debug_mode']:
            all_user_list = pos_df['user_id'].unique()[:1000]
        else:
            all_user_list = pos_df['user_id'].unique()
        all_item_list, item_num = pos_df['item_id'].unique(), pos_df['item_id'].nunique()
        id2category = dict(zip(pos_df['item_id'], pos_df['categories']))
        id2category.update({0: 0})
        for user in tqdm(all_user_list, desc="构建数据集"):
            for i in range(len(pos_dict[user]) - 5, len(pos_dict[user]) - 1):
                user_id_list.append(user)
                item_target_id_list.append(pos_dict[user][i + 1])
                item_target_category_list.append(id2category[pos_dict[user][i + 1]])
                label_list.append(1)
                if i < max_seq:
                    item_history_seq_id_list.append(pos_dict[user][:i] + [0] * (max_seq - i))
                else:
                    item_history_seq_id_list.append(pos_dict[user][i - max_seq:i])
                item_history_seq_category_list.append([id2category[item] for item in item_history_seq_id_list[-1]])

                for i in range(neg_sample_ratio):
                    user_id_list.append(user)
                    label_list.append(0)
                    item_history_seq_id_list.append(item_history_seq_id_list[-1])
                    item_history_seq_category_list.append(item_history_seq_category_list[-1])

                    temp_item_index = random.randint(0, item_num - 1)
                    while all_item_list[temp_item_index] in pos_dict[user]:
                        temp_item_index = random.randint(0, item_num - 1)
                    item_target_id_list.append(all_item_list[temp_item_index])
                    item_target_category_list.append(id2category[all_item_list[temp_item_index]])
        data = {
            'user_id': user_id_list,
            'item_target_id': item_target_id_list,
            'item_target_category': item_target_category_list,
            'item_history_seq_id': item_history_seq_id_list,
            'item_history_seq_category': item_history_seq_category_list,
            'label': label_list,
        }
        logger.info("数据集构建完成,其中正样本数目为：{}，负样本数目为：{}".format(label_list.count(1), label_list.count(0)))
        data = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
        return data

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pos_df, pos_dict = self.load_data()
        data = self.construct_data(pos_df, pos_dict, 20, self.config["neg_sample_ratio"])
        train_df = data[:int(len(data) * self.config["train_ratio"])].reset_index(drop=True)
        valid_df = data[int(len(data) * self.config["train_ratio"]):int(len(data) * (self.config["train_ratio"] + self.config["valid_ratio"]))].reset_index(drop=True)
        test_df = data[int(len(data) * (self.config["train_ratio"] + self.config["valid_ratio"])):].reset_index(drop=True)
        logger.info("数据集切分完成，其中训练集、验证集、测试集的比例为：{:.1f}:{:.1f}:{:.1f}".format(self.config["train_ratio"], self.config["valid_ratio"], 1 - self.config["train_ratio"] - self.config["valid_ratio"]))
        return train_df, valid_df, test_df

    def get_dataset(self, data: pd.DataFrame, enc_dict: dict = None) -> Dataset:
        dataset = get_amazon_dataset(data, self.config, enc_dict)
        return dataset

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


class ProcessData():

    def __init__(self, config: dict) -> None:
        self.config = config
        self.data = self.get_data()

    def get_data(self):
        if self.config["data"] == "criteo":
            data = CriteoProcessData(config=self.config)
        elif self.config["data"] == "movielens":
            data = MovieLenProcessData(config=self.config)
        elif self.config["data"] == "amazon":
            data = AmazonProcessData(config=self.config)
        else:
            raise ValueError("数据集错误")
        return data

    def data_process(self):
        return self.data.data_process()
