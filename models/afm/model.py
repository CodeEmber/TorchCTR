'''
Author       : wyx-hhhh
Date         : 2024-03-04
LastEditTime : 2024-05-27
Description  : 
'''
from itertools import combinations
from typing import List
import torch
import torch.nn as nn
from layers.embedding_layer import EmbeddingLayer
from layers.logistic_regression_layer import LogRegLayer
from managers.logger_manager import logger
from layers.mlp_layer import MLPLayer
from utils.torch_utils import get_dnn_input_dim, get_linear_input, set_activation, set_loss_func


class FMPlus(nn.Module):

    def __init__(self, embedding_dim=32, attention_factor=32, dropout_rate=0):
        super(FMPlus, self).__init__()
        self.attention_factor = attention_factor
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.attention_net = nn.Sequential(
            nn.Linear(self.embedding_dim, self.attention_factor),
            nn.ReLU(),
            nn.Linear(self.attention_factor, 1, bias=False),
            nn.Softmax(dim=1),
        )
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, feature_emb_list):
        pair_feature = []
        for vi, vj in combinations(feature_emb_list, 2):
            pair_feature.append(vi * vj)
        pair_feature = torch.cat(pair_feature, dim=1)  # [batch,num_pair:<(n-1)*n/2>,emb]
        attention_weight = self.attention_net(pair_feature)  # [batch,num_pair,1]
        if self.dropout_rate > 0:
            attention_weight = self.dropout(attention_weight)
        pair_feature = torch.sum(pair_feature * attention_weight, dim=1)  # [batch,emb]
        pair_feature = torch.sum(pair_feature, dim=-1).unsqueeze(-1)  # [batch,1]
        return pair_feature


#AFM 模型
class AFM(nn.Module):

    def __init__(
        self,
        embedding_dim=10,
        attention_factor=32,
        dropout_rate=0,
        l2_reg=0.1,
        loss_fun='bce',
        enc_dict=None,
    ):
        super(AFM, self).__init__()

        self.embedding_dim = embedding_dim
        self.attention_factor = attention_factor
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.loss_fun = set_loss_func(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)

        self.fm = FMPlus(
            embedding_dim=embedding_dim,
            attention_factor=attention_factor,
            dropout_rate=dropout_rate,
        )  #二阶
        self.lr = LogRegLayer(enc_dict=enc_dict)  #一阶

    def get_l2_reg_loss(self, ):
        attention_W = self.fm.attention_net[0].weight
        reg_loss = torch.sum(self.l2_reg * attention_W**2)
        return reg_loss

    def forward(self, data):
        sparse_embedding = self.embedding_layer(data)

        # FM
        lr_logit = self.lr(data)  #一阶交叉
        fm_logit = self.fm(sparse_embedding)  #二阶交叉

        #输出
        y_pred = torch.sigmoid(lr_logit + fm_logit)
        loss = self.loss_fun(y_pred, data["label"].view(-1, 1).float())
        loss += self.get_l2_reg_loss()
        output_dict = {'y_pred': y_pred, 'loss': loss}
        return output_dict
