'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2023-10-28
Description  : 
'''
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(self, enc_dict: dict = None, embedding_dim: int = None):
        super(EmbeddingLayer, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()
        self.emb_feature = []
        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col: nn.Embedding(
                    self.enc_dict[col]['vocab_size'],
                    self.embedding_dim,
                )})

    def forward(self, x):
        feature_emb_list = []
        for col in self.emb_feature:
            inp = x[col].long().view(-1, 1)
            feature_emb_list.append(self.embedding_layer[col](inp))
        return feature_emb_list
