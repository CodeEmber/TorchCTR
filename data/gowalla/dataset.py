'''
Author       : wyx-hhhh
Date         : 2024-04-08
LastEditTime : 2024-06-28
Description  : 
'''
import random
import pandas as pd
from torch.utils.data import Dataset
import dgl
import torch
from dgl import graph
import numpy as np


class GowallaDataset(Dataset):

    def __init__(self, df: pd.DataFrame, config: dict, phase: str = 'train'):
        self.config = config
        self.df = df
        self.phase = phase
        self.user_num = self.config["user_num"]
        self.item_num = self.config["item_num"]
        self.generate_grouped_data()
        if phase == 'train':
            self.encode_data()

    def encode_data(self):
        self.data = dict()
        self.data['user_id'] = torch.Tensor(np.array(self.df['user_id'].values)).long()
        self.data['item_id'] = torch.Tensor(np.array(self.df['item_id'].values)).long()

    def generate_grouped_data(self):
        self.grouped_data = self.df.groupby('user_id')['item_id'].apply(list).to_dict()

    def generate_graph(self):
        if self.config.get("add_self_loop", True):
            src_node_list = torch.cat([self.data['user_id'], self.data['item_id'] + self.user_num, torch.arange(self.user_num + self.item_num)], axis=0)
            dst_node_list = torch.cat([self.data['item_id'] + self.user_num, self.data['user_id'], torch.arange(self.user_num + self.item_num)], axis=0)
        else:
            # 不添加自环可能会出现部分节点没有边的情况，然后在之后Embedding匹配过程中可能会出现问题
            src_node_list = torch.cat([self.data['user_id'], self.data['item_id'] + self.user_num], axis=0)
            dst_node_list = torch.cat([self.data['item_id'] + self.user_num, self.data['user_id']], axis=0)
        g = graph((src_node_list, dst_node_list))

        src_degree = g.out_degrees().float()
        norm = torch.pow(src_degree, -0.5).unsqueeze(1)  #compute norm
        g.ndata['norm'] = norm  #节点粒度的norm

        edge_weight = norm[src_node_list] * norm[dst_node_list]  #计算边的权重
        g.edata['edge_weight'] = edge_weight  # 边的权重
        return g

    def __getitem__(self, index) -> dict:
        data = dict()
        if self.phase == "train":
            random_index = random.randint(0, self.user_num - 1)
            while random_index in self.grouped_data[self.df['user_id'].iloc[index]]:
                random_index = random.randint(0, self.user_num - 1)
            neg_item_id = torch.Tensor([random_index]).squeeze().long()
            data['neg_item_id'] = neg_item_id
        data['user_id'] = self.data['user_id'][index]
        data['item_id'] = self.data['item_id'][index]
        return data

    def __len__(self):
        if self.phase == 'train':
            return len(self.df)
        else:
            return self.df['user_id'].nunique()


def get_gowalla_dataset(data: str, config: dict, phase: dict) -> GowallaDataset:
    dataset = GowallaDataset(data, config, phase)
    return dataset
