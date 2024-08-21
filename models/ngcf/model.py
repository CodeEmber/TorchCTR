'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):

    def __init__(self, g, config):
        super(NGCF, self).__init__()
        self.config = config
        self.g = g
        self.user_num = self.config['user_num']
        self.item_num = self.config['item_num']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_units = self.config['hidden_units']
        self.message_dropout_ratio = self.config['message_dropout_ratio']
        self.node_dropout_ratio = self.config['node_dropout_ratio']
        self.decay = self.config['decay']
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.f = nn.Sigmoid()
        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()
        """
        *********************************************************
        Get sparse adj.
        """
        # self.sparse_g = self._convert_sp_mat_to_sp_tensor(self.g).to(self.device)
        self.sparse_g = g

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.embedding_dim))), 'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.embedding_dim)))})

        weight_dict = nn.ParameterDict()
        layers = [self.embedding_dim] + self.hidden_units
        for k in range(len(self.hidden_units)):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse_coo_tensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users)**2 + torch.norm(pos_items)**2 + torch.norm(neg_items)**2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def get_users_rating(self, users, final_user_embedding, final_item_embedding):
        users = users.cpu()
        user_embedding = final_user_embedding[users.long()]
        item_embedding = final_item_embedding
        rating = self.f(torch.matmul(user_embedding, item_embedding.t()))
        return rating

    def forward(self, data, is_train=True):

        A_hat = self.sparse_dropout(self.sparse_g, self.node_dropout_ratio, self.sparse_g._nnz())

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.hidden_units)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            ego_embeddings = nn.Dropout(self.message_dropout_ratio)(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.user_num, :]
        i_g_embeddings = all_embeddings[self.user_num:, :]
        """
        *********************************************************
        look up.
        """
        # u_g_embeddings = u_g_embeddings[users, :]
        # pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        # neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        # return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

        output_dict = dict()

        if is_train:
            user_embedding = u_g_embeddings[data['user_id'], :]
            pos_item_embedding = i_g_embeddings[data['item_id'], :]
            neg_item_embedding = i_g_embeddings[data['neg_item_id'], :]
            batch_loss, batch_mf_loss, batch_emb_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
            output_dict['loss'] = batch_loss
            output_dict['mf_loss'] = batch_mf_loss
            output_dict['emb_loss'] = batch_emb_loss
        else:
            output_dict['user_embedding'] = u_g_embeddings
            output_dict['item_embedding'] = i_g_embeddings
        return output_dict
