'''
Author       : wyx-hhhh
Date         : 2024-06-26
LastEditTime : 2024-08-21
Description  : LightGCN model implementation.
'''

import torch
import torch.nn as nn
import dgl.function as fn
import dgl


class LightGCN(nn.Module):

    def __init__(self, config: dict, g):
        super(LightGCN, self).__init__()
        self.config = config
        self.g = g
        self.user_num = self.config['user_num']
        self.item_num = self.config['item_num']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_units = self.config['hidden_units']
        self.lmbd = self.config['lmbd']
        self.keep_prob = self.config['keep_prob']

        # Embedding layers for users and items
        self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_dim)

        # Hidden units configuration
        self.hidden_units = [self.embedding_dim] + self.hidden_units
        self.f = nn.Sigmoid()

        # Initialize weights
        self.init_weights()

        # Cache for embeddings
        self.final_user_embedding = None
        self.final_item_embedding = None

    def init_weights(self):
        nn.init.normal_(self.user_embedding_layer.weight, std=0.1)
        nn.init.normal_(self.item_embedding_layer.weight, std=0.1)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) < keep_prob  # Use less than for dropout
        index = index[random_index]
        values = values[random_index] / keep_prob  # Scale values
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.config['is_split']:
            graph = []
            for g in self.g:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.g, keep_prob)
        return graph

    def computer(self):
        user_embedding = self.user_embedding_layer.weight  # [user_num, embedding_dim]
        item_embedding = self.item_embedding_layer.weight  # [item_num, embedding_dim]

        all_emb = torch.cat([user_embedding, item_embedding])  # Concatenate user and item embeddings
        embs = [all_emb]

        # Apply dropout if necessary
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.g
        else:
            g_droped = self.g

        # Propagate embeddings through the layers
        for _ in range(len(self.hidden_units) - 1):
            if self.config['is_split']:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)  # Stack embeddings
        light_out = torch.mean(embs, dim=1)  # Average embeddings
        final_user_embedding, final_item_embedding = torch.split(light_out, [self.user_num, self.item_num])

        # Cache the embeddings for test phase
        if not self.training:
            self.final_user_embedding = final_user_embedding
            self.final_item_embedding = final_item_embedding

        return final_user_embedding, final_item_embedding

    def bpr_loss(self, embedding_dict):
        user_embedding = embedding_dict["user_embedding"]
        pos_item_embedding = embedding_dict["pos_item_embedding"]
        neg_item_embedding = embedding_dict["neg_item_embedding"]
        user_embedding_ego = embedding_dict["user_embedding_ego"]
        pos_item_embedding_ego = embedding_dict["pos_item_embedding_ego"]
        neg_item_embedding_ego = embedding_dict["neg_item_embedding_ego"]

        # Regularization loss
        # reg_loss = (torch.norm(user_embedding_ego)**2 + torch.norm(pos_item_embedding_ego)**2 + torch.norm(neg_item_embedding_ego)**2) / 2 / user_embedding.shape[0]
        reg_loss = (1 / 2) * (user_embedding_ego.norm(2).pow(2) + pos_item_embedding_ego.norm(2).pow(2) + neg_item_embedding_ego.norm(2).pow(2)) / user_embedding.shape[0]

        # BPR loss calculation
        pos_scores = torch.mul(user_embedding, pos_item_embedding)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_embedding, neg_item_embedding)
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # pos_scores = (user_embedding * pos_item_embedding).sum(dim=1)
        # neg_scores = (user_embedding * neg_item_embedding).sum(dim=1)
        # bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        return bpr_loss + self.lmbd * reg_loss

    def get_embedding(self, data, final_user_embedding, final_item_embedding):
        embedding_dict = dict()
        embedding_dict["user_embedding"] = final_user_embedding[data['user_id']]
        embedding_dict["pos_item_embedding"] = final_item_embedding[data['item_id']]
        embedding_dict["neg_item_embedding"] = final_item_embedding[data['neg_item_id']]
        embedding_dict["user_embedding_ego"] = self.user_embedding_layer(data['user_id'])
        embedding_dict["pos_item_embedding_ego"] = self.item_embedding_layer(data['item_id'])
        embedding_dict["neg_item_embedding_ego"] = self.item_embedding_layer(data['neg_item_id'])
        return embedding_dict

    def get_users_rating(self, users, final_user_embedding, final_item_embedding):
        user_embedding = final_user_embedding[users.long()]
        item_embedding = final_item_embedding
        rating = self.f(torch.matmul(user_embedding, item_embedding.t()))
        return rating

    def forward(self, data, is_train=True):
        final_user_embedding, final_item_embedding = self.computer()

        output_dict = dict()
        if is_train:
            embedding_dict = self.get_embedding(data, final_user_embedding, final_item_embedding)
            loss = self.bpr_loss(embedding_dict)
            output_dict['loss'] = loss
        else:
            output_dict['user_embedding'] = self.final_user_embedding
            output_dict['item_embedding'] = self.final_item_embedding

        return output_dict
