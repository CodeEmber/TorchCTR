import torch
import torch.nn as nn
import torch.nn.functional as F

from base.graph_recommender import GraphRecommender
from utils.torch_loss import BPRLoss, L2RegLoss


class NGCF(GraphRecommender):

    def __init__(self, train_config):
        super(NGCF, self).__init__(train_config)
        self.model = NGCFModel(self.data_dict["graph_data"], self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])


class NGCFModel(nn.Module):

    def __init__(self, g, config):
        super(NGCFModel, self).__init__()
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
        self.embedding_dict, self.weight_dict = self.init_weight()
        self.sparse_g = g
        self.bpr_loss = BPRLoss()
        self.reg_loss = L2RegLoss(self.decay, self.batch_size)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.embedding_dim))), 'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.embedding_dim)))}).to(self.config["device"])
        weight_dict = nn.ParameterDict().to(self.config["device"])
        layers = [self.embedding_dim] + self.hidden_units
        for k in range(len(self.hidden_units)):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1]))).to(self.config["device"])})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1]))).to(self.config["device"])})

            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1]))).to(self.config["device"])})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1]))).to(self.config["device"])})

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

    def forward(self, data, is_train=True):

        A_hat = self.sparse_dropout(self.sparse_g, self.node_dropout_ratio, self.sparse_g._nnz())
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.hidden_units)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.message_dropout_ratio)(ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.user_num, :]
        i_g_embeddings = all_embeddings[self.user_num:, :]
        output_dict = dict()

        if is_train:
            user_embedding = u_g_embeddings[data['user_id'], :]
            pos_item_embedding = i_g_embeddings[data['item_id'], :]
            neg_item_embedding = i_g_embeddings[data['neg_item_id'], :]
            user_embedding_ego = self.embedding_dict['user_emb'][data['user_id'], :]
            pos_item_embedding_ego = self.embedding_dict['item_emb'][data['item_id'], :]
            neg_item_embedding_ego = self.embedding_dict['item_emb'][data['neg_item_id'], :]
            batch_bpr_loss = self.bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
            # batch_reg_loss = self.reg_loss(user_embedding, pos_item_embedding, neg_item_embedding)
            batch_reg_loss = self.reg_loss(user_embedding_ego, pos_item_embedding_ego, neg_item_embedding_ego)
            output_dict['loss'] = batch_bpr_loss + batch_reg_loss
            output_dict['batch_bpr_loss'] = batch_bpr_loss
            output_dict['batch_reg_loss'] = batch_reg_loss
        else:
            output_dict['user_embedding'] = u_g_embeddings
            output_dict['item_embedding'] = i_g_embeddings
        return output_dict
