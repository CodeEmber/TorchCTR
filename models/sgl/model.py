'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-09-06
Description  : 
'''
from typing import List
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from utils.torch_loss import BPRLoss, InfoNCELoss, L2RegLoss
from utils.utilities import get_file_path
from utils.middleware import time_middleware
from utils.utilities import get_values_by_keys


class SGL(GraphRecommender):

    def __init__(self, train_config):
        super(SGL, self).__init__(train_config)
        self.model = SGLModel(self.config, self.data_dict["graph_data"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

    def train(self):
        self.model.generation_augmented_graph()
        super(SGL, self).train()


class SGLModel(nn.Module):

    def __init__(
        self,
        config: dict,
        g,
    ):
        super(SGLModel, self).__init__()
        self.config = config
        self.g = g
        self.user_num = self.config['user_num']
        self.item_num = self.config['item_num']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_units = self.config['hidden_units']
        self.num_layers = len(self.hidden_units)
        self.lmbd_reg = self.config['lmbd_reg']
        self.lmbd_ssl = self.config['lmbd_ssl']
        self.aug_type = self.config['aug_type']
        self.dropout_ratio = self.config['dropout_ratio']
        self.cl_tau = self.config['cl_tau']
        self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_dim).to(self.config["device"])
        self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_dim).to(self.config["device"])

        self.hidden_units = [self.embedding_dim] + self.hidden_units
        self.f = nn.Sigmoid()
        self.bpr_loss = BPRLoss()
        self.reg_loss = L2RegLoss(self.lmbd_reg, self.config['batch_size'])
        self.infonce_loss = InfoNCELoss(self.cl_tau)
        self.init_generate_sub_graph()
        self.apply(self.init_weights)

    def init_generate_sub_graph(self):
        train_data = self.config["data_dict"]["train_df"][["user_id", "item_id"]].values
        self.n_nodes = self.user_num + self.item_num
        self.users_np = np.array([i[0] for i in train_data])
        self.items_np = np.array([i[1] for i in train_data])

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def node_dropout(self, dropout_ratio):
        drop_user_idx = np.random.choice(self.user_num, size=int(self.user_num * dropout_ratio), replace=False)
        drop_item_idx = np.random.choice(self.item_num, size=int(self.item_num * dropout_ratio), replace=False)
        indicator_user = np.ones(self.user_num, dtype=np.float32)
        indicator_item = np.ones(self.item_num, dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        R = sp.csr_matrix((np.ones_like(self.users_np, dtype=np.float32), (self.users_np, self.items_np)), shape=(self.user_num, self.item_num))
        R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
        (user_np_keep, item_np_keep) = R_prime.nonzero()
        ratings_keep = R_prime.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.user_num)), shape=(self.n_nodes, self.n_nodes))
        return tmp_adj

    def edge_dropout(self, dropout_ratio):
        keep_idx = np.random.choice(len(self.users_np), size=int(len(self.users_np) * (1 - dropout_ratio)), replace=False)
        user_np = np.array(self.users_np)[keep_idx]
        item_np = np.array(self.items_np)[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(self.n_nodes, self.n_nodes))
        return tmp_adj

    @time_middleware('生成增强图')
    def create_sub_graph(self):
        if self.aug_type in ['ED', 'RW']:
            tmp_adj = self.edge_dropout(self.dropout_ratio)
        elif self.aug_type == 'ND':
            tmp_adj = self.node_dropout(self.dropout_ratio)
        else:
            raise ValueError('Augmentation type must be one of ED, ND, RW')
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        rowsum = np.where(rowsum == 0, 1e-10, rowsum)
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        coo = adj_matrix.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        new_g = torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()
        new_g = new_g.to(self.config['device'])
        return new_g

    def generation_augmented_graph(self):
        if self.aug_type in ['ED', 'ND']:
            self.sub_graph1 = [self.create_sub_graph()] * self.num_layers
            self.sub_graph2 = [self.create_sub_graph()] * self.num_layers
        elif self.aug_type == 'RW':
            self.sub_graph1 = [self.create_sub_graph() for _ in range(self.num_layers)]
            self.sub_graph2 = [self.create_sub_graph() for _ in range(self.num_layers)]
        else:
            raise ValueError('Augmentation type must be one of ED, ND, RW')

    def computer(self, g: List[torch.sparse.FloatTensor] | torch.sparse.FloatTensor):
        user_embedding = self.user_embedding_layer.weight
        item_embedding = self.item_embedding_layer.weight

        all_emb = torch.cat([user_embedding, item_embedding])
        embs = [all_emb]

        for i in range(self.num_layers):
            if isinstance(g, list):
                all_emb = torch.sparse.mm(g[i], all_emb)
            else:
                all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        final_user_embedding, final_item_embedding = torch.split(light_out, [self.user_num, self.item_num])
        return final_user_embedding, final_item_embedding

    def create_infonce_loss(self, data, sub_embedding_dict):
        u_idx = data['user_id']
        i_idx = data['item_id']
        user_cl_loss = self.infonce_loss(sub_embedding_dict["final_user_embedding1"][u_idx], sub_embedding_dict["final_user_embedding2"][u_idx])
        item_cl_loss = self.infonce_loss(sub_embedding_dict["final_item_embedding1"][i_idx], sub_embedding_dict["final_item_embedding2"][i_idx])
        return user_cl_loss + item_cl_loss

    def total_loss(self, data, embedding_dict, sub_embedding_dict):
        bpr_loss = self.bpr_loss(embedding_dict["user_embedding"], embedding_dict["pos_item_embedding"], embedding_dict["neg_item_embedding"])
        reg_loss = self.reg_loss(embedding_dict["user_embedding_ego"], embedding_dict["pos_item_embedding_ego"], embedding_dict["neg_item_embedding_ego"])
        ssl_loss = self.create_infonce_loss(data, sub_embedding_dict)
        return bpr_loss + ssl_loss * self.lmbd_ssl + reg_loss * self.lmbd_reg

    def get_embedding(self, data, final_user_embedding, final_item_embedding):
        embedding_dict = dict()
        embedding_dict["user_embedding"] = final_user_embedding[data['user_id']]
        embedding_dict["pos_item_embedding"] = final_item_embedding[data['item_id']]
        embedding_dict["neg_item_embedding"] = final_item_embedding[data['neg_item_id']]
        embedding_dict["user_embedding_ego"] = self.user_embedding_layer(data['user_id'])
        embedding_dict["pos_item_embedding_ego"] = self.item_embedding_layer(data['item_id'])
        embedding_dict["neg_item_embedding_ego"] = self.item_embedding_layer(data['neg_item_id'])
        return embedding_dict

    def forward(self, data, is_train=True):
        final_user_embedding, final_item_embedding = self.computer(self.g)
        output_dict = dict()

        if is_train:
            final_user_embedding1, final_item_embedding1 = self.computer(self.sub_graph1)
            final_user_embedding2, final_item_embedding2 = self.computer(self.sub_graph2)
            sub_embedding_dict = {
                "final_user_embedding1": final_user_embedding1,
                "final_item_embedding1": final_item_embedding1,
                "final_user_embedding2": final_user_embedding2,
                "final_item_embedding2": final_item_embedding2,
            }
            embedding_dict = self.get_embedding(
                data,
                final_user_embedding,
                final_item_embedding,
            )

            loss = self.total_loss(data, embedding_dict, sub_embedding_dict)
            output_dict['loss'] = loss
        else:
            output_dict['user_embedding'] = final_user_embedding
            output_dict['item_embedding'] = final_item_embedding
        return output_dict
