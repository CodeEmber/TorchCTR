'''
Author       : wyx-hhhh
Date         : 2024-03-04
LastEditTime : 2024-03-25
Description  : 
'''
from typing import List
from matplotlib import category
import torch
import torch.nn as nn
from managers.logger_manager import logger
from layers.mlp_layer import MLPLayer
from utils.torch_utils import get_dnn_input_dim, get_linear_input, set_activation, set_loss_func


class EmbeddingLayer(nn.Module):

    def __init__(self, enc_dict: dict, embedding_dim: int):
        super(EmbeddingLayer, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim

        self.item_cols = []
        self.history_cols = []
        self.user_cols = []

        self.embedding_dict = nn.ModuleDict()
        for col in self.enc_dict:
            if 'vocab_size' in self.enc_dict[col]:
                if self.enc_dict[col]['type'] == 'user':
                    self.embedding_dict[col] = nn.Embedding(
                        num_embeddings=self.enc_dict[col]['vocab_size'],
                        embedding_dim=self.embedding_dim,
                        padding_idx=0,
                    )
                    self.user_cols.append(col)
                elif self.enc_dict[col]['type'] == 'item':
                    self.embedding_dict[col] = nn.Embedding(
                        num_embeddings=self.enc_dict[col]['vocab_size'],
                        embedding_dim=self.embedding_dim,
                        padding_idx=0,
                    )
                    self.item_cols.append(col)
                else:
                    raise ValueError(f"Unknown type: {self.enc_dict[col]['type']}")
            elif 'share_with' in self.enc_dict[col]:
                self.history_cols.append(col)

    def forward(self, data):
        item_emb_list = []
        user_emb_list = []
        history_emb_list = []
        for col in self.item_cols:
            item_emb_list.append(self.embedding_dict[col](data[col]))
        item_emb = torch.cat(item_emb_list, dim=1).squeeze(1)
        for col in self.user_cols:
            user_emb_list.append(self.embedding_dict[col](data[col]))
        user_emb = torch.cat(user_emb_list, dim=1)

        for col in self.history_cols:
            history_emb_list.append(self.embedding_dict[self.enc_dict[col]['share_with']](data[col]))
        history_emb = torch.cat(history_emb_list, dim=-1)
        return user_emb, item_emb, history_emb


class Dice(nn.Module):

    def __init__(self, input_dim: int, alpha: float = 0., eps=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float))

    def forward(self, x):
        px = torch.sigmoid(self.bn(x))
        output = self.alpha * (1 - px) * x + px * x  # (batch_size, input_dim)
        return output


class DINAttentionLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        attention_units: List[int],
        hidden_activation: str = "Dice",
        dice_alpha=0,
    ):
        super(DINAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        if hidden_activation == "Dice":
            self.hidden_activation = Dice(
                self.embedding_dim * 4,
                attention_units,
                dice_alpha,
            )
        else:
            self.hidden_activation = set_activation(hidden_activation)
        self.lr = nn.Linear(self.embedding_dim * 4, 1)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)

    def get_mask(self, history_seq):
        return (history_seq > 0).sum(dim=-1)

    def forward(self, item_emb, history_emb):
        # item_emb: (batch_size, embedding_dim)
        # history_emb: (batch_size, history_len, embedding_dim)
        mask = self.get_mask(history_emb)  # (batch_size, history_len)
        history_seq_len = history_emb.size(1)

        item_emb = item_emb.unsqueeze(1).expand(-1, history_emb.size(1), -1)  # (batch_size, history_len, embedding_dim)
        attention_input = torch.cat(
            [item_emb, history_emb, item_emb - history_emb, item_emb * history_emb],
            dim=-1,
        )  # (batch_size, history_len, embedding_dim * 4)

        attention_input = attention_input.view(-1, self.embedding_dim * 4)  # (batch_size * history_len, embedding_dim * 4)
        attention_weight = self.lr(self.hidden_activation(attention_input))  # (batch_size * history_len, 1)
        attention_weight = attention_weight.view(-1, history_seq_len)  # (batch_size, history_len)
        attention_weight = attention_weight * mask.float()
        output = (attention_weight.unsqueeze(-1) * history_emb).sum(dim=1)  # (batch_size, embedding_dim)
        return output


class DIN(nn.Module):

    def __init__(
        self,
        enc_dict: dict = None,
        hidden_units: List[int] = [64, 32, 16],
        attention_units: List[int] = [32],
        embedding_dim: int = 16,
        loss_func: str = "bce",
    ):
        super(DIN, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.attention_units = attention_units
        self.loss_func = set_loss_func(loss_func)
        self.count_map = self.enc_dict['count_map']

        self.embedding_layer = EmbeddingLayer(self.enc_dict, self.embedding_dim)
        self.user_label_num, self.item_label_num = self.get_feature_num()
        self.din_attention_layer = DINAttentionLayer(
            self.embedding_dim * self.item_label_num,
            self.attention_units,
        )
        self.mlp_layer = MLPLayer(
            input_dim=self.embedding_dim * (2 * self.item_label_num + self.user_label_num),
            hidden_units=self.hidden_units,
            output_dim=1,
        )

    def get_feature_num(self):
        user_label_num = 0
        item_label_num = 0
        for col in self.enc_dict:
            if 'vocab_size' in self.enc_dict[col]:
                if self.enc_dict[col]['type'] == 'user':
                    user_label_num += 1
                elif self.enc_dict[col]['type'] == 'item':
                    item_label_num += 1
        return user_label_num, item_label_num

    def MBA_Reg(self, data, col):
        feature_id = torch.unique(data[col])
        feature_emb_list = []
        for id in feature_id:
            feature_emb_list.append(self.embedding_layer.embedding_dict[col](id)**2 / self.count_map[col][int(id.detach().cpu().numpy())])
        feature_emb = torch.cat(feature_emb_list, dim=0).mean()
        return feature_emb

    def lr_reg(self, data):
        user_mba_l2 = self.MBA_Reg(data, 'user_id')
        item_mba_l2 = self.MBA_Reg(data, 'item_target_id')
        category_mba_l2 = self.MBA_Reg(data, 'item_target_category')
        return user_mba_l2 + item_mba_l2 + category_mba_l2

    def forward(self, data):
        # (batch_size, embedding_dim*user_label_num)，(batch_size, embedding_dim*item_label_num)，(batch_size, history_len, embedding_dim*item_label_num)
        user_emb, item_emb, history_emb = self.embedding_layer(data)
        attention_output = self.din_attention_layer(item_emb, history_emb)
        mlp_input = torch.cat([user_emb, item_emb, attention_output], dim=1)
        mlp_output = self.mlp_layer(mlp_input)
        y_pred = mlp_output.sigmoid()
        loss = self.loss_func(y_pred, data["label"].float().view(-1, 1)) + 0.2 * self.lr_reg(data)
        output_dict = {"y_pred": y_pred, "loss": loss}
        return output_dict
