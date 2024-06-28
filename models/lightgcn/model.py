'''
Author       : wyx-hhhh
Date         : 2024-06-26
LastEditTime : 2024-06-28
Description  : 
'''
import torch
import torch.nn as nn
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
        g: dgl.DGLGraph,
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
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def custom_loss(self, user_embedding, pos_item_embedding, neg_item_embedding):
        pos_scores = (user_embedding * pos_item_embedding).sum(dim=1)
        neg_scores = (user_embedding * neg_item_embedding).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        regularizer = (torch.norm(user_embedding)**2 + torch.norm(pos_item_embedding)**2 + torch.norm(neg_item_embedding)**2) / 2
        return loss + self.lmbd * regularizer / user_embedding.shape[0]

    def forward(self, data, is_train=True):
        user_embedding = self.user_embedding_layer.weight  # [user_num, embedding_dim]
        item_embedding = self.item_embedding_layer.weight  # [item_num, embedding_dim]

        final_user_embedding_list = [user_embedding]  # [user_embedding, layer1_user_embedding, layer2_user_embedding, ...]
        final_item_embedding_list = [item_embedding]  # [item_embedding, layer1_item_embedding, layer2_item_embedding, ...]

        for _ in range(len(self.hidden_units)):
            hidden = self.gcn_layer(self.g, user_embedding, item_embedding)
            user_embedding, item_embedding = torch.split(hidden, [self.user_num, self.item_num], dim=0)
            final_user_embedding_list.append(user_embedding)
            final_item_embedding_list.append(item_embedding)

        final_user_embedding = torch.mean(torch.stack(final_user_embedding_list, 1), 1)
        final_item_embedding = torch.mean(torch.stack(final_item_embedding_list, 1), 1)

        output_dict = dict()

        if is_train:
            user_embedding = final_user_embedding[data['user_id'], :]
            pos_item_embedding = final_item_embedding[data['item_id'], :]
            neg_item_embedding = final_item_embedding[data['neg_item_id'], :]
            loss = self.custom_loss(user_embedding, pos_item_embedding, neg_item_embedding)
            output_dict['loss'] = loss
        else:
            output_dict['user_embedding'] = final_user_embedding
            output_dict['item_embedding'] = final_item_embedding
        return output_dict
