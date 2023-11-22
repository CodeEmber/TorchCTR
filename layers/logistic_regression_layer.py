'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2023-10-28
Description  : 
'''
import torch
import torch.nn as nn

from layers.embedding_layer import EmbeddingLayer
from utils.torch_utils import get_dnn_input_dim, get_linear_input


class LogRegLayer(nn.Module):

    def __init__(self, enc_dict: dict = None):
        super(LogRegLayer, self).__init__()
        self.enc_dict = enc_dict
        self.emb_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=1)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, 1)
        self.linear = nn.Linear(self.dnn_input_dim, 1)

    def forward(self, x):
        sparse_emb_list = self.emb_layer(x)
        sparse_emb_list = torch.stack(sparse_emb_list, dim=1).flatten(1)
        dense_input = get_linear_input(self.enc_dict, x)
        dnn_input = torch.cat((sparse_emb_list, dense_input), axis=1)
        out = self.linear(dnn_input)
        return out
