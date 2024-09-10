'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-09-09
Description  : 
'''
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.graph_recommender import GraphRecommender
from utils.torch_loss import BPRLoss, InfoNCELoss, L2RegLoss
from utils.utilities import get_values_by_keys

from typing import List
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utilities import get_file_path
from utils.middleware import time_middleware
from utils.utilities import get_values_by_keys


class SimGCL(GraphRecommender):

    def __init__(self, train_config):
        super(SimGCL, self).__init__(train_config)
        self.model = SimGCLModel(self.config, self.data_dict["graph_data"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

    def train(self):
        super(SimGCL, self).train()


class SimGCLModel(nn.Module):

    def __init__(
        self,
        config: dict,
        g,
    ):
        super(SimGCLModel, self).__init__()
        self.config = config
        self.g = g
        self.user_num = self.config['user_num']
        self.item_num = self.config['item_num']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_units = self.config['hidden_units']
        self.num_layers = len(self.hidden_units)
        self.lmbd_reg = self.config['lmbd_reg']
        self.lmbd_ssl = self.config['lmbd_ssl']
        self.cl_tau = self.config['cl_tau']
        self.eps = self.config['eps']
        self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_dim).to(self.config["device"])
        self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_dim).to(self.config["device"])

        self.hidden_units = [self.embedding_dim] + self.hidden_units
        self.f = nn.Sigmoid()
        self.bpr_loss = BPRLoss()
        self.reg_loss = L2RegLoss(self.lmbd_reg, self.config["batch_size"])
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

    def computer(self, g: torch.sparse.FloatTensor, perturbed=False):
        user_embedding = self.user_embedding_layer.weight
        item_embedding = self.item_embedding_layer.weight

        all_emb = torch.cat([user_embedding, item_embedding])
        embs = []

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(g, all_emb)
            if perturbed:
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(torch.randn_like(all_emb), dim=-1) * self.eps
            else:
                all_emb = all_emb
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

        final_user_embedding, final_item_embedding = self.computer(self.g, perturbed=False)

        output_dict = dict()

        if is_train:
            final_user_embedding1, final_item_embedding1 = self.computer(self.g, perturbed=True)
            final_user_embedding2, final_item_embedding2 = self.computer(self.g, perturbed=True)
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
