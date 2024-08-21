'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-08-16
Description  : 
'''
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utilities import get_values_by_keys


class SGL(nn.Module):

    def __init__(
        self,
        config: dict,
        g,
    ):
        super(SGL, self).__init__()
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
        self.ssl_tau = self.config['ssl_tau']
        self.generation_augmented_graph()
        self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_dim)

        self.hidden_units = [self.embedding_dim] + self.hidden_units

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def node_dropout(self, g, dropout_ratio):
        keep_prob = 1.0 - dropout_ratio
        size = g.size()
        indices = g.indices()
        values = g.values()

        num_nodes = size[0]
        keep_mask = torch.ones(num_nodes, dtype=torch.bool)
        drop_mask = torch.rand(num_nodes) < dropout_ratio
        keep_mask[drop_mask] = False

        keep_edges_mask = keep_mask[indices[0]] & keep_mask[indices[1]]
        new_indices = indices[:, keep_edges_mask]
        new_values = values[keep_edges_mask] / keep_prob

        updated_adj = torch.sparse_coo_tensor(new_indices, new_values, size)

        return updated_adj

    def edge_dropout(self, g, dropout_ratio):
        keep_prob = 1.0 - dropout_ratio
        size = g.size()
        index = g.indices().t()
        values = g.values()

        random_index = torch.rand(len(values)) < keep_prob
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g

    def generation_augmented_graph(self):
        if self.aug_type == 'ED':
            self.sub_graph1 = [self.edge_dropout(self.g, self.dropout_ratio)] * self.num_layers
            self.sub_graph2 = [self.edge_dropout(self.g, self.dropout_ratio)] * self.num_layers
        elif self.aug_type == 'ND':
            self.sub_graph1 = [self.node_dropout(self.g, self.dropout_ratio)] * self.num_layers
            self.sub_graph2 = [self.node_dropout(self.g, self.dropout_ratio)] * self.num_layers
        elif self.aug_type == 'RW':
            self.sub_graph1 = [self.edge_dropout(self.g, self.dropout_ratio) for _ in range(self.num_layers)]
            self.sub_graph2 = [self.edge_dropout(self.g, self.dropout_ratio) for _ in range(self.num_layers)]
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

    # def get_augmented_embedding(self, sub_graph):
    #     user_embedding = self.user_embedding_layer.weight  # [user_num, embedding_dim]
    #     item_embedding = self.item_embedding_layer.weight  # [item_num, embedding_dim]
    #     all_emb = torch.cat([user_embedding, item_embedding])
    #     embs = [all_emb]

    #     for i in range(self.num_layers):
    #         user_embedding, item_embedding = self.computer(sub_graph[i])
    #         all_emb = torch.cat([user_embedding, item_embedding])
    #         embs.append(all_emb)
    #     embs = torch.stack(embs, dim=1)
    #     light_out = torch.mean(embs, dim=1)
    #     final_user_embedding, final_item_embedding = torch.split(light_out, [self.user_num, self.item_num])
    #     return final_user_embedding, final_item_embedding

    def create_bpr_loss(self, embedding_dict):
        user_embedding, pos_item_embedding, neg_item_embedding = embedding_dict["user_embedding"], embedding_dict["pos_item_embedding"], embedding_dict["neg_item_embedding"]
        pos_scores = (user_embedding * pos_item_embedding).sum(dim=1)
        neg_scores = (user_embedding * neg_item_embedding).sum(dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return bpr_loss

    def create_regularization_loss(self, embedding_dict):
        user_embedding_ego, pos_item_embedding_ego, neg_item_embedding_ego = embedding_dict["user_embedding_ego"], embedding_dict["pos_item_embedding_ego"], embedding_dict["neg_item_embedding_ego"]
        reg_loss = (torch.norm(user_embedding_ego)**2 + torch.norm(pos_item_embedding_ego)**2 + torch.norm(neg_item_embedding_ego)**2) / (2 * user_embedding_ego.shape[0])
        return reg_loss

    def create_ssl_loss(self, data, sub_embedding_dict):
        user_embedding1, item_embedding1, user_embedding2, item_embedding2 = get_values_by_keys(
            data=sub_embedding_dict,
            keys=[
                "final_user_embedding1",
                "final_item_embedding1",
                "final_user_embedding2",
                "final_item_embedding2",
            ],
        )
        user_embeddings1 = F.normalize(user_embedding1, dim=1)
        item_embeddings1 = F.normalize(item_embedding1, dim=1)
        user_embeddings2 = F.normalize(user_embedding2, dim=1)
        item_embeddings2 = F.normalize(item_embedding2, dim=1)

        user_embs1 = F.embedding(data['user_id'], user_embeddings1)
        item_embs1 = F.embedding(data['item_id'], item_embeddings1)
        user_embs2 = F.embedding(data['user_id'], user_embeddings2)
        item_embs2 = F.embedding(data['item_id'], item_embeddings2)

        pos_ratings_user = torch.sum(user_embs1 * user_embs2, dim=1)
        pos_ratings_item = torch.sum(item_embs1 * item_embs2, dim=1)
        tot_ratings_user = torch.matmul(user_embs1, torch.transpose(user_embeddings2, 0, 1))
        tot_ratings_item = torch.matmul(item_embs1, torch.transpose(item_embeddings2, 0, 1))

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_tau, dim=1)
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_tau, dim=1)
        infonce_loss = torch.sum(clogits_user + clogits_item)
        return infonce_loss

    def total_loss(self, data, embedding_dict, sub_embedding_dict):
        bpr_loss = self.create_bpr_loss(embedding_dict)
        reg_loss = self.create_regularization_loss(embedding_dict)
        ssl_loss = self.create_ssl_loss(data, sub_embedding_dict)
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
        final_user_embedding1, final_item_embedding1 = self.computer(self.sub_graph1)
        final_user_embedding2, final_item_embedding2 = self.computer(self.sub_graph2)
        sub_embedding_dict = {
            "final_user_embedding1": final_user_embedding1,
            "final_item_embedding1": final_item_embedding1,
            "final_user_embedding2": final_user_embedding2,
            "final_item_embedding2": final_item_embedding2,
        }
        output_dict = dict()

        if is_train:
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
