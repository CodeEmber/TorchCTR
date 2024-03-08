'''
Author       : wyx-hhhh
Date         : 2024-03-04
LastEditTime : 2024-03-05
Description  : 
'''
from typing import List
import torch
import torch.nn as nn
from layers import EmbeddingLayer
from layers.mlp_layer import MLPLayer
from utils.torch_utils import get_dnn_input_dim, get_linear_input, set_activation, set_loss_func


class GMF(nn.Module):

    def __init__(self, embedding_dim):
        super(GMF, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, user_emb, item_emb):
        out = self.fc(user_emb * item_emb).sigmoid()
        return out


class NCF(nn.Module):

    def __init__(
        self,
        enc_dict: dict = None,
        embedding_dim1: int = 16,
        embedding_dim2: int = 32,
        hidden_units: List[int] = [64, 32, 16],
        loss_func: str = "bce",
    ):
        """初始化NCF
    
            Args:
                enc_dict (dict, optional): 编码后的特征字典. Defaults to None.
                embedding_dim1 (int, optional): GMF部分embedding维度. Defaults to 16.
                embedding_dim2 (int, optional): MLP部分embedding维度. Defaults to 32.
                hidden_units (List[int], optional): 隐藏层维度. Defaults to [64, 32, 16].
                loss_func (str, optional): 损失函数. Defaults to "bce".
                
            """
        super(NCF, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim1 = embedding_dim1
        self.embedding_dim2 = embedding_dim2
        self.loss_func = set_loss_func(loss_func)
        # GMF
        self.user_emb_layer1 = nn.Embedding(self.enc_dict['user_id']['vocab_size'], self.embedding_dim1)
        self.item_emb_layer1 = nn.Embedding(self.enc_dict['item_id']['vocab_size'], self.embedding_dim1)
        # MLP
        self.user_emb_layer2 = nn.Embedding(self.enc_dict['user_id']['vocab_size'], self.embedding_dim2)
        self.item_emb_layer2 = nn.Embedding(self.enc_dict['item_id']['vocab_size'], self.embedding_dim2)
        self.gmf = GMF(self.embedding_dim1)
        self.mlp = MLPLayer(
            input_dim=self.embedding_dim2 * 2,
            hidden_units=hidden_units,
            hidden_activation="relu",
            dropout_rate=0,
            batch_norm=True,
            use_bias=True,
        )
        self.fc = nn.Linear(self.embedding_dim1 + hidden_units[-1], 1)

    def forward(self, x):
        user_emb1 = self.user_emb_layer1(x["user_id"])
        item_emb1 = self.item_emb_layer1(x["item_id"])
        user_emb2 = self.user_emb_layer2(x["user_id"])
        item_emb2 = self.item_emb_layer2(x["item_id"])
        gmf_out = self.gmf(user_emb1, item_emb1)
        mlp_input = torch.cat([user_emb2, item_emb2], dim=1)
        mlp_out = self.mlp(mlp_input)
        final_input = torch.cat([gmf_out, mlp_out], dim=1)
        out = self.fc(final_input).sigmoid()
        loss = self.loss_func(out, x["label"].float().view(-1, 1))
        output_dict = {"y_pred": out, "loss": loss}
        return output_dict
