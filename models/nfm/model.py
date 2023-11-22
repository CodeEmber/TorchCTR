'''
Author       : wyx-hhhh
Date         : 2023-10-31
LastEditTime : 2023-11-07
Description  : 
'''
from typing import List
import torch
import torch.nn as nn
from layers import EmbeddingLayer, LogRegLayer
from layers.mlp_layer import MLPLayer
from utils.torch_utils import get_dnn_input_dim, get_feature_num, get_linear_input, set_activation, set_loss_func


class BiInteractionLayer(nn.Module):

    def __init__(self):
        super(BiInteractionLayer, self).__init__()

    def forward(self, feature_emb):
        feature_emb = torch.stack(feature_emb, dim=1).squeeze(2)
        sum_of_square = torch.sum(feature_emb, dim=1)**2
        square_of_sum = torch.sum(feature_emb**2, dim=1)
        return 0.5 * (sum_of_square - square_of_sum)


class NFM(nn.Module):

    def __init__(
        self,
        embedding_dim=10,
        hidden_units=[64, 64, 64],
        bi_dropout=0.5,
        loss_func='bce',
        enc_dict=None,
    ):
        super(NFM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.bi_dropout = bi_dropout
        self.loss_func = set_loss_func(loss_func)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=enc_dict, embedding_dim=embedding_dim)
        self.lr = LogRegLayer(enc_dict=enc_dict)
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(bi_dropout)
        self.bi_interaction = BiInteractionLayer()
        _, dense_feature_num = get_feature_num(enc_dict)
        self.dnn_input_dim = self.embedding_dim + dense_feature_num
        self.dnn = MLPLayer(
            input_dim=self.dnn_input_dim,
            output_dim=1,
            hidden_units=self.hidden_units,
            hidden_activation='relu',
            final_activation='relu',
            dropout_rate=0,
        )

    def forward(self, x):
        sparse_embedding = self.embedding_layer(x)
        dense_input = get_linear_input(self.enc_dict, x)
        lr = self.lr(x)

        pooling_output = self.bi_interaction(sparse_embedding)
        if self.bi_dropout > 0:
            pooling_output = self.dropout(pooling_output)
        dnn_input = torch.cat([pooling_output, dense_input], dim=1)
        dnn_output = self.dnn(dnn_input)

        y_pred = torch.sigmoid(lr + dnn_output)
        loss = self.loss_func(y_pred, x["label"].float().view(-1, 1))
        return {'y_pred': y_pred, 'loss': loss}
