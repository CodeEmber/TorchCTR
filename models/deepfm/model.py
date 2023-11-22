'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2023-10-28
Description  : 
'''
import torch
import torch.nn as nn
from typing import List

from layers import EmbeddingLayer, LogRegLayer, MLPLayer
from utils.torch_utils import get_dnn_input_dim, get_linear_input, set_loss_func


class FMLayer(nn.Module):

    def __init__(self):
        super(FMLayer, self).__init__()

    def forward(self, feature_emb):
        feature_emb = tuple(feature_emb)
        feature_emb = torch.stack(feature_emb, dim=1)
        sum_of_square = torch.sum(feature_emb, dim=1)**2
        square_of_sum = torch.sum(feature_emb**2, dim=1)
        fm_out = (sum_of_square - square_of_sum) * 0.5
        return torch.sum(fm_out, dim=-1).view(-1, 1)


class DeepFM(nn.Module):

    def __init__(
        self,
        enc_dict: dict = None,
        embedding_dim: int = 10,
        loss_func: str = "bce",
    ):
        super(DeepFM, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.loss_func = set_loss_func(loss_func)
        self.emb_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        self.lr = LogRegLayer(enc_dict=self.enc_dict)
        self.fm = FMLayer()
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLPLayer(
            input_dim=self.dnn_input_dim,
            output_dim=1,
            hidden_units=[64, 64, 64],
            hidden_activation="relu",
            final_activation="relu",
            dropout_rate=0.5,
            batch_norm=True,
            use_bias=True,
        )

    def forward(self, x):
        sparse_emb_list = self.emb_layer(x)
        sparse_emb_list = torch.stack(sparse_emb_list, dim=1).flatten(start_dim=1)
        dense_input = get_linear_input(self.enc_dict, x)

        fm_out = self.fm(sparse_emb_list)
        lr_out = self.lr(x)
        fm_out = fm_out + lr_out

        dnn_input = torch.cat((sparse_emb_list, dense_input), dim=1)
        dnn_out = self.dnn(dnn_input)

        y_pred = torch.sigmoid(fm_out + dnn_out)
        loss = self.loss_func(y_pred, x["label"].view(-1, 1).float())
        output = {"y_pred": y_pred, "loss": loss}
        return output
