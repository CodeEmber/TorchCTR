'''
Author       : wyx-hhhh
Date         : 2024-04-08
LastEditTime : 2024-09-10
Description  : 
'''
import random
from re import T
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from sympy import E, false
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numba import jit
from torch.utils.data import Dataset
from utils.utilities import get_file_path
from utils.middleware import time_middleware
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from cppimport import imp_from_filepath
    import sys
    sys.path.append(get_file_path(["data", "sources"]))
    path = get_file_path(["data", "sources", "sampling.cpp"])
    sampling = imp_from_filepath(path)
    is_cython = True
except:
    is_cython = False


class GeneralDataset(Dataset):

    def __init__(self, df: pd.DataFrame, config: dict, phase: str = 'train'):
        self.config = config
        self.df = df
        self.phase = phase
        self.user_num = self.config["user_num"]
        self.item_num = self.config["item_num"]
        self.generate_grouped_data()
        self.Graph = None
        self.user_id = []
        self.item_id = []
        self.neg_samples = []
        self.UserItemNet = csr_matrix((np.ones(len(self.df)), (self.df['user_id'], self.df['item_id'])), shape=(self.user_num, self.item_num))
        if phase == 'train':
            self.negative_sampling()
            self.encode_data()

    def negative_sampling(self):
        if is_cython:
            self.negative_sampling_cython()
        else:
            self.config["logger"].info("Cython negative sampling failed, using python negative sampling")
            self.negative_sampling_python()

    @time_middleware("使用python负采样完成")
    def negative_sampling_python(self):
        if self.config.get('data_dict', {}).get('pred_gd') and self.config.get("is_control_negative_sampling", False):
            train_grouped_data = self.config["data_dict"]["train_grouped_data"]
            test_grouped_data = self.config["data_dict"]["test_grouped_data"]
            pred_gd = self.config["data_dict"]["pred_gd"]
            for user in train_grouped_data.keys():
                all_item_list = list(range(self.item_num))
                neg_item_list = []
                pos_items = train_grouped_data[user]
                if len(pos_items) == 0:
                    continue
                neg_item_list.extend(list(set(all_item_list) - set(pos_items)))
                a = list(set(pred_gd[user]) - set(test_grouped_data[user]) - set(pos_items))
                neg_item_list.extend(a * 3)
                neg_item = random.sample(neg_item_list, len(pos_items))
                for i in range(len(pos_items)):
                    self.user_id.append(user)
                    self.item_id.append(pos_items[i])
                    self.neg_samples.append(neg_item[i])
        else:
            user = self.grouped_data.keys()
            for u in user:
                pos_items = self.grouped_data[u]
                if len(pos_items) == 0:
                    continue
                for i in range(len(pos_items)):
                    self.user_id.append(u)
                    self.item_id.append(pos_items[i])
                    while True:
                        neg_item = np.random.randint(0, self.item_num)
                        if neg_item not in pos_items:
                            break
                    self.neg_samples.append(neg_item)

        # users = np.random.randint(0, self.user_num, len(self.df))
        # for _, user in enumerate(users):
        #     pos_items = self.grouped_data[user]
        #     if len(pos_items) == 0:
        #         continue
        #     pos_item = np.random.choice(pos_items)
        #     while True:
        #         neg_item = np.random.randint(0, self.item_num)
        #         if neg_item not in pos_items:
        #             break
        #     self.neg_samples.append(neg_item)
        #     self.user_id.append(user)
        #     self.item_id.append(pos_item)

    @time_middleware("使用c++负采样完成")
    def negative_sampling_cython(self):
        # sampling.set_seed(self.config.get("seed", 2024))
        if self.config.get('data_dict', {}).get('pred_gd') and self.config.get("is_control_negative_sampling", False):
            all_data = sampling.negative_sampling_control(
                self.config["data_dict"]["train_grouped_data"],
                self.config["data_dict"]["test_grouped_data"],
                self.config["data_dict"]["pred_gd"],
                self.config.get("neg_num", 1),
                self.item_num,
                self.config.get("error_item_ratio", 300),
            )
        else:
            all_data = sampling.negative_sampling(
                self.grouped_data,
                self.config.get("neg_num", 1),
                self.item_num,
            )
        self.user_id = all_data[:, 0]
        self.item_id = all_data[:, 1]
        self.neg_samples = all_data[:, 2]

    def encode_data(self):
        self.data = dict()
        if self.phase == 'train':
            self.data['user_id'] = torch.Tensor(np.array(self.user_id)).long()
            self.data['item_id'] = torch.Tensor(np.array(self.item_id)).long()
            self.data['neg_item_id'] = torch.Tensor(np.array(self.neg_samples)).long()
        else:
            self.data['user_id'] = torch.Tensor(np.array(self.df['user_id'].values)).long()
            self.data['item_id'] = torch.Tensor(np.array(self.df['item_id'].values)).long()

    def reset_negative_sampling(self):
        self.user_id = []
        self.item_id = []
        self.neg_samples = []
        self.negative_sampling()
        self.encode_data()

    def generate_grouped_data(self):
        self.grouped_data = self.df.groupby('user_id')['item_id'].apply(list).to_dict()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    @time_middleware("稀疏矩阵生成成功")
    def generate_graph(self):
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(get_file_path(self.config['s_pre_adj_mat']))
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                # 遍历用户并填充邻接矩阵
                for i in range(0, self.user_num, 30000):
                    # 计算当前处理的用户数量
                    end_index = min(i + 30000, self.user_num)
                    adj_mat[i:end_index, self.user_num:] = R[i:end_index]
                    adj_mat[self.user_num:, i:end_index] = R[i:end_index].T
                # adj_mat[:self.user_num, self.user_num:] = R
                # adj_mat[self.user_num:, :self.user_num] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                sp.save_npz(get_file_path(self.config['s_pre_adj_mat']), norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.config['device'])
        return self.Graph

    def __getitem__(self, index) -> dict:
        data = dict()
        if self.phase == "train":
            data['neg_item_id'] = self.data['neg_item_id'][index]
        data['user_id'] = self.data['user_id'][index]
        data['item_id'] = self.data['item_id'][index]
        return data

    def __len__(self):
        if self.phase == 'train':
            return len(self.data['user_id'])
        else:
            return self.df['user_id'].nunique()


def get_dataset(data: str, config: dict, phase: dict) -> GeneralDataset:
    dataset = GeneralDataset(data, config, phase)
    return dataset


if __name__ == "__main__":
    train_grouped_data = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
    }
    test_grouped_data = {
        0: [3, 4, 5],
        1: [6, 7, 8],
        2: [0, 1, 2],
    }
    pred_gd = {
        0: [8, 4, 5],
        1: [6, 7, 8],
        2: [0, 1, 2],
    }
    neg_num = 1
    item_num = 9
    error_item_ratio = 3
    a = sampling.negative_sampling_control(
        train_grouped_data,
        test_grouped_data,
        pred_gd,
        neg_num,
        item_num,
        error_item_ratio,
    )
    print(a)
