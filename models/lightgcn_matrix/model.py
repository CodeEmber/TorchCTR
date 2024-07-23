'''
Author       : wyx-hhhh
Date         : 2024-06-26
LastEditTime : 2024-07-19
Description  : 
'''
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import dgl.function as fn
import dgl


class LightGCNConv(nn.Module):

    def __init__(self):
        super(LightGCNConv, self).__init__()

    def forward(
        self,
        g: dgl.DGLGraph,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor,
    ):
        g.ndata['h'] = torch.cat([user_embedding, item_embedding], dim=0)  # [user_num + item_num, embedding_dim]
        g.update_all(fn.u_mul_e('h', 'edge_weight', 'm'), fn.sum('m', 'h'))
        h = nn.functional.normalize(g.ndata['h'], p=2, dim=1)
        return h


class LightGCN(nn.Module):

    def __init__(
        self,
        config: dict,
        g,
    ):
        super(LightGCN, self).__init__()
        self.config = config
        self.g = g
        self.user_num = self.config['user_num']
        self.item_num = self.config['item_num']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_units = self.config['hidden_units']
        self.lmbd = self.config['lmbd']

        self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_dim)

        self.hidden_units = [self.embedding_dim] + self.hidden_units
        self.gcn_layer = LightGCNConv()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(tensor=module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.config['is_split']:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        user_embedding = self.user_embedding_layer.weight  # [user_num, embedding_dim]
        item_embedding = self.item_embedding_layer.weight  # [item_num, embedding_dim]

        all_emb = torch.cat([user_embedding, item_embedding])
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.g
        else:
            g_droped = self.g

        for _ in range(len(self.hidden_units)):
            if self.config['is_split']:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        final_user_embedding, final_item_embedding = torch.split(light_out, [self.user_num, self.item_num])
        return final_user_embedding, final_item_embedding

    def bpr_loss(self, embedding_dict):
        user_embedding, pos_item_embedding, neg_item_embedding = embedding_dict["user_embedding"], embedding_dict["pos_item_embedding"], embedding_dict["neg_item_embedding"]
        user_embedding_ego, pos_item_embedding_ego, neg_item_embedding_ego = embedding_dict["user_embedding_ego"], embedding_dict["pos_item_embedding_ego"], embedding_dict["neg_item_embedding_ego"]
        reg_loss = (torch.norm(user_embedding_ego)**2 + torch.norm(pos_item_embedding_ego)**2 + torch.norm(neg_item_embedding_ego)**2) / 2 / user_embedding.shape[0]
        pos_scores = (user_embedding * pos_item_embedding).sum(dim=1)
        neg_scores = (user_embedding * neg_item_embedding).sum(dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return bpr_loss + self.lmbd * reg_loss

    def get_embedding(self, data):
        final_user_embedding, final_item_embedding = self.computer()
        embedding_dict = dict()
        embedding_dict["user_embedding"] = final_user_embedding[data['user_id']]
        embedding_dict["pos_item_embedding"] = final_item_embedding[data['item_id']]
        embedding_dict["neg_item_embedding"] = final_item_embedding[data['neg_item_id']]
        embedding_dict["user_embedding_ego"] = self.user_embedding_layer(data['user_id'])
        embedding_dict["pos_item_embedding_ego"] = self.item_embedding_layer(data['item_id'])
        embedding_dict["neg_item_embedding_ego"] = self.item_embedding_layer(data['neg_item_id'])
        return embedding_dict

    def forward(self, data, is_train=True):
        final_user_embedding, final_item_embedding = self.computer()

        output_dict = dict()
        if is_train:
            embedding_dict = self.get_embedding(data)
            loss = self.bpr_loss(embedding_dict)
            output_dict['loss'] = loss
        else:
            output_dict['user_embedding'] = final_user_embedding
            output_dict['item_embedding'] = final_item_embedding
        return output_dict
