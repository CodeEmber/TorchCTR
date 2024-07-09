'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-07-09
Description  : 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl import DGLGraph, graph


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


class SGL(nn.Module):

    def __init__(
        self,
        config: dict,
        g: dgl.DGLGraph,
    ):
        super(SGL, self).__init__()
        self.config = config
        self.g = g
        self.user_num = self.config['user_num']
        self.item_num = self.config['item_num']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_units = self.config['hidden_units']
        self.num_layers = len(self.hidden_units)
        self.lmbd = self.config['lmbd']
        self.aug_type = self.config['aug_type']
        self.node_dropout_num = self.config['node_dropout_num']
        self.edge_dropout_num = self.config['edge_dropout_num']
        self.ssl_tau = self.config['ssl_tau']
        self.ssl_weight = self.config['ssl_weight']
        self.generation_augmented_graph()
        self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_dim)

        self.hidden_units = [self.embedding_dim] + self.hidden_units
        self.gcn_layer = LightGCNConv()

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def calculate_graph(self, g: dgl.DGLGraph):
        src_node_list, dst_node_list = g.edges()
        src_degree = g.out_degrees().float()
        norm = torch.pow(src_degree, -0.5).unsqueeze(1)
        g.ndata['norm'] = norm
        edge_weight = norm[src_node_list] * norm[dst_node_list]
        g.edata['edge_weight'] = edge_weight
        return g

    def node_dropout(self, g: dgl.DGLGraph, node_dropout_num: float):
        num_nodes = g.num_nodes()
        nodes_list = g.nodes()
        row, col = g.edges()
        mask = torch.rand(num_nodes, device=g.device) >= node_dropout_num

        keep_nodes = nodes_list[mask]
        filter_row, filter_col = row[keep_nodes], col[keep_nodes]

        g1 = graph((filter_row, filter_col), num_nodes=g.num_nodes())
        g1 = dgl.add_self_loop(g1)

        return self.calculate_graph(g1)

    def edge_dropout(self, g: dgl.DGLGraph, edge_dropout_num: float):
        row, col = g.edges()
        mask = torch.rand(row.size(0), device=row.device) >= edge_dropout_num
        filter_row, filter_col = row[mask], col[mask]
        g1 = graph((filter_row, filter_col), num_nodes=g.num_nodes())
        g1 = dgl.add_self_loop(g1)

        return self.calculate_graph(g1)

    def generation_augmented_graph(self):
        if self.aug_type == 'ED':
            self.sub_graph1 = [self.edge_dropout(self.g, self.edge_dropout_num)] * self.num_layers
            self.sub_graph2 = [self.edge_dropout(self.g, self.edge_dropout_num)] * self.num_layers
        elif self.aug_type == 'ND':
            self.sub_graph1 = [self.node_dropout(self.g, self.node_dropout_num)] * self.num_layers
            self.sub_graph2 = [self.node_dropout(self.g, self.node_dropout_num)] * self.num_layers
        elif self.aug_type == 'RW':
            self.sub_graph1 = [self.edge_dropout(self.g, self.edge_dropout) for _ in range(self.num_layers)]
            self.sub_graph2 = [self.edge_dropout(self.g, self.edge_dropout) for _ in range(self.num_layers)]
        else:
            raise ValueError('Augmentation type must be one of ED, ND, RW')

    def get_augmented_embedding(self, sub_graph):
        user_embedding = self.user_embedding_layer.weight  # [user_num, embedding_dim]
        item_embedding = self.item_embedding_layer.weight  # [item_num, embedding_dim]
        final_user_embedding_list = [user_embedding]
        final_item_embedding_list = [item_embedding]

        for i in range(self.num_layers):
            hidden = self.gcn_layer(sub_graph[i], user_embedding, item_embedding)
            user_embedding, item_embedding = torch.split(hidden, [self.user_num, self.item_num], dim=0)
            final_user_embedding_list.append(user_embedding)
            final_item_embedding_list.append(item_embedding)

        final_user_embedding = torch.mean(torch.stack(final_user_embedding_list, 1), 1)
        final_item_embedding = torch.mean(torch.stack(final_item_embedding_list, 1), 1)
        return final_user_embedding, final_item_embedding

    def create_bpr_loss(self, user_embedding, pos_item_embedding, neg_item_embedding):
        pos_scores = (user_embedding * pos_item_embedding).sum(dim=1)
        neg_scores = (user_embedding * neg_item_embedding).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        regularizer = (torch.norm(user_embedding)**2 + torch.norm(pos_item_embedding)**2 + torch.norm(neg_item_embedding)**2) / 2
        return loss + self.lmbd * regularizer / user_embedding.shape[0]

    def create_ssl_loss(self, data):
        user_sub1, item_sub1 = self.get_augmented_embedding(self.sub_graph1)
        user_sub2, item_sub2 = self.get_augmented_embedding(self.sub_graph2)

        u_emd1 = F.normalize(user_sub1[data['user_id']], dim=1)
        u_emd2 = F.normalize(user_sub2[data['user_id']], dim=1)  # [batch,emb]
        all_user2 = F.normalize(user_sub2, dim=1)  # [num_user,emb]
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)  # [batch,num_user]
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[data['item_id']], dim=1)
        i_emd2 = F.normalize(item_sub2[data['item_id']], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

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
            bpr_loss = self.create_bpr_loss(
                user_embedding,
                pos_item_embedding,
                neg_item_embedding,
            )
            ssl_loss = self.create_ssl_loss(data, )
            loss = bpr_loss + ssl_loss
            output_dict['loss'] = loss
        else:
            output_dict['user_embedding'] = final_user_embedding
            output_dict['item_embedding'] = final_item_embedding
        return output_dict
