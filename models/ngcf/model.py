'''
Author       : wyx-hhhh
Date         : 2024-03-04
LastEditTime : 2024-06-28
Description  : 
'''
import torch
import torch.nn as nn
import dgl.function as fn
from dgl import graph
import dgl


class NGCFLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super(NGCFLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.W1 = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.W2 = nn.Linear(self.in_dim, self.out_dim, bias=False)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(self.dropout)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)

    def message_func(self, edges):
        edge_feature = self.W1(edges.src['h']) + self.W2(edges.src['h'] * edges.dst['h'])
        edge_feature = edge_feature * (edges.src['norm'] * edges.dst['norm'])
        return {'e': edge_feature}

    def forward(self, g, user_embedding, item_embedding):
        g.ndata['h'] = torch.cat([user_embedding, item_embedding], dim=0)  # [user_num + item_num, embedding_dim]
        g.update_all(self.message_func, fn.sum(msg='e', out='m'))
        g.ndata['m'] = g.ndata['m'] + self.W1(g.ndata['h'])
        h = self.leaky_relu(g.ndata['m'])
        h = self.dropout(h)
        h = nn.functional.normalize(h, p=2, dim=1)
        return h  # [user_num + item_num, embedding_dim]


class NGCF(nn.Module):

    def __init__(
        self,
        config: dict,
        g: dgl.DGLGraph,
    ):
        super(NGCF, self).__init__()
        self.config = config
        self.g = g
        self.user_num = self.config['user_num']
        self.item_num = self.config['item_num']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_units = self.config['hidden_units']
        self.message_dropout = self.config['message_dropout']
        self.node_dropout = self.config['node_dropout']
        self.lmbd = self.config['lmbd']

        self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_dim)

        self.ngcf_layers = nn.ModuleList()
        self.hidden_units = [self.embedding_dim] + self.hidden_units

        for i in range(len(self.hidden_units) - 1):
            self.ngcf_layers.append(NGCFLayer(self.hidden_units[i], self.hidden_units[i + 1], self.message_dropout))

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def node_dropout_fn(
        self,
        g: dgl.DGLGraph,
        node_dropout: float,
    ):
        """
        Implement node dropout and compute node norms and edge weights.
        
        Args:
            g (DGLGraph): The input graph.
            node_dropout (float): The node dropout rate.
        
        Returns:
            DGLGraph: The modified graph with node dropout and updated edge weights.
        """
        # Concatenate source and destination nodes
        src_node_list = torch.cat([g.srcnodes(), g.dstnodes()], dim=0)
        dst_node_list = torch.cat([g.dstnodes(), g.srcnodes()], dim=0)

        # Compute node degrees
        src_degree = g.out_degrees().float()

        # Compute node norms
        norm = torch.pow(src_degree, -0.5).unsqueeze(1)

        # Compute edge weights
        edge_weight = norm[src_node_list] * norm[dst_node_list]
        edge_weight = edge_weight.squeeze()

        # Apply node dropout
        keep_mask = torch.rand(len(src_node_list)) >= node_dropout
        src_node_list = src_node_list[keep_mask]
        dst_node_list = dst_node_list[keep_mask]
        edge_weight = edge_weight[keep_mask]

        # Create the modified graph
        g = dgl.graph((src_node_list, dst_node_list), num_nodes=g.num_nodes())
        g.edata['edge_weight'] = edge_weight

        return g

    def custom_loss(self, user_embedding, pos_item_embedding, neg_item_embedding):
        pos_scores = (user_embedding * pos_item_embedding).sum(dim=1)
        neg_scores = (user_embedding * neg_item_embedding).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        regularizer = (torch.norm(user_embedding)**2 + torch.norm(pos_item_embedding)**2 + torch.norm(neg_item_embedding)**2) / 2
        return loss + self.lmbd * regularizer / user_embedding.shape[0]

    def forward(self, data, is_train=True):
        user_embedding = self.user_embedding_layer.weight  # [user_num, embedding_dim]
        item_embedding = self.item_embedding_layer.weight  # [item_num, embedding_dim]

        final_user_embedding = [user_embedding]  # [user_embedding, layer1_user_embedding, layer2_user_embedding, ...]
        final_item_embedding = [item_embedding]  # [item_embedding, layer1_item_embedding, layer2_item_embedding, ...]

        g = self.node_dropout_fn(self.g, self.node_dropout)

        for ngcf_layer in self.ngcf_layers:
            hidden = ngcf_layer(g, user_embedding, item_embedding)
            user_embedding, item_embedding = torch.split(hidden, [self.user_num, self.item_num], dim=0)
            final_user_embedding.append(user_embedding)
            final_item_embedding.append(item_embedding)

        output_dict = dict()

        final_user_embedding = torch.cat(final_user_embedding, dim=1)
        final_item_embedding = torch.cat(final_item_embedding, dim=1)
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
