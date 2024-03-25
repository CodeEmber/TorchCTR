'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-03-22
Description  : 
'''
import random

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from data.movielens.dataset import get_movie_len_dataset
from managers.logger_manager import logger
from data.base_process_data import BaseProcessData
from utils.file_utils import get_file_path
from data.criteo.dataset import get_criteo_dataset


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
        return train_df, pd.DataFrame(), test_df

    def get_dataset(self, data: pd.DataFrame, enc_dict: dict = None) -> Dataset:
        dataset = get_movie_len_dataset(data, self.config, enc_dict)
        return dataset
