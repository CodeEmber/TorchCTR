'''
Author       : wyx-hhhh
Date         : 2024-04-08
LastEditTime : 2024-08-16
Description  : 
'''
import random
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from sympy import E, false
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torch.utils.data import Dataset
from utils.file_utils import get_file_path
from utils.middleware import time_middleware

try:
    from cppimport import imp_from_filepath
    import sys
    sys.path.append(get_file_path(["data", "sources"]))
    path = get_file_path(["data", "sources", "sampling.cpp"])
    sampling = imp_from_filepath(path)
    is_cython = True
except:
    is_cython = False


class YelpDataset(Dataset):

    def __init__(self, df: pd.DataFrame, config: dict, phase: str = 'train'):
        self.config = config
        self.df = df
        self.phase = phase
        self.user_num = self.config["user_num"]
        self.item_num = self.config["item_num"]
        self.generate_grouped_data()
        self.Graph = None
        self.neg_samples = []

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
        for user_id in self.df['user_id']:
            neg_item_id = random.randint(0, self.item_num - 1)
            while neg_item_id in self.grouped_data[user_id]:
                neg_item_id = random.randint(0, self.item_num - 1)
            self.neg_samples.append(neg_item_id)

    @time_middleware("使用c++负采样完成")
    def negative_sampling_cython(self):
        sampling.set_seed(self.config.get("seed", 2024))
        self.neg_samples = sampling.negative_sampling(self.user_num, self.item_num, len(self.df), self.grouped_data)

    def encode_data(self):
        self.data = dict()
        self.data['user_id'] = torch.Tensor(np.array(self.df['user_id'].values)).long()
        self.data['item_id'] = torch.Tensor(np.array(self.df['item_id'].values)).long()
        if self.phase == 'train':
            self.data['neg_item_id'] = torch.Tensor(np.array(self.neg_samples)).long()

    def generate_grouped_data(self):
        self.grouped_data = self.df.groupby('user_id')['item_id'].apply(list).to_dict()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.user_num + self.item_num) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.user_num + self.item_num
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.config["device"]))
        return A_fold

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
                adj_mat[:self.user_num, self.user_num:] = R
                adj_mat[self.user_num:, :self.user_num] = R.T
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

            if self.config.get("is_split", false):
                self.Graph = self._split_A_hat(norm_adj)
            else:
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
            return len(self.df)
        else:
            return self.df['user_id'].nunique()


def get_gowalla_matrix_dataset(data: str, config: dict, phase: dict) -> YelpDataset:
    dataset = YelpDataset(data, config, phase)
    return dataset
