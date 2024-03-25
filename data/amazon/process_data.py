'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-03-25
Description  : 
'''
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from managers.logger_manager import logger
from data.base_process_data import BaseProcessData
from utils.file_utils import get_file_path
from data.amazon.dataset import get_amazon_dataset


class AmazonProcessData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def get_enc_dict(self, pos_df) -> dict:
        """本数据集对应的enc_dict

        Args:
            pos_df (DataFram): 处理后的数据集
        """
        self.enc_dict = {}
        self.enc_dict["user_id"] = {'vocab_size': pos_df['user_id'].nunique() + 1, 'type': 'user'}
        self.enc_dict["item_target_id"] = {'vocab_size': pos_df['item_id'].nunique() + 1, 'type': 'item'}
        self.enc_dict["item_target_category"] = {'vocab_size': pos_df['categories'].nunique() + 1, 'type': 'item'}
        self.enc_dict["item_history_seq_id"] = {'share_with': 'item_target_id', 'type': 'item'}
        self.enc_dict["item_history_seq_category"] = {'share_with': 'item_target_category', 'type': 'item'}
        self.enc_dict["count_map"] = {
            'user_id': pos_df['user_id'].value_counts().to_dict(),
            'item_target_id': pos_df['item_id'].value_counts().to_dict(),
            'item_target_category': pos_df['categories'].value_counts().to_dict(),
        }

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
        self.get_enc_dict(pos_df)
        data = self.construct_data(pos_df, pos_dict, 20, self.config["neg_sample_ratio"])
        train_df = data[:int(len(data) * self.config["train_ratio"])].reset_index(drop=True)
        valid_df = data[int(len(data) * self.config["train_ratio"]):int(len(data) * (self.config["train_ratio"] + self.config["valid_ratio"]))].reset_index(drop=True)
        test_df = data[int(len(data) * (self.config["train_ratio"] + self.config["valid_ratio"])):].reset_index(drop=True)
        logger.info("数据集切分完成，其中训练集、验证集、测试集的比例为：{:.1f}:{:.1f}:{:.1f}".format(self.config["train_ratio"], self.config["valid_ratio"], 1 - self.config["train_ratio"] - self.config["valid_ratio"]))
        return train_df, valid_df, test_df

    def get_dataset(self, data: pd.DataFrame, enc_dict: dict = None) -> Dataset:
        dataset = get_amazon_dataset(data, self.config, self.enc_dict)
        return dataset
