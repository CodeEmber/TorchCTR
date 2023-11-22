from typing import List
import torch
import torch.nn as nn
from itertools import combinations
from layers import EmbeddingLayer, LogRegLayer, MLPLayer
from utils.torch_utils import get_dnn_input_dim, get_feature_num, get_linear_input, set_activation, set_loss_func


class SENETLayer(nn.Module):

    def __init__(self, num_fields: int, reduction_ratio: int = 3):
        """初始化SENETLayer

        Args:
            input_dim (int): 输入维度
            reduction_ratio (int, optional): 降维比例. Defaults to 3.
        """
        super(SENETLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.ReLU(),
        )

    def forward(self, feature_emb):
        if len(feature_emb.shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(feature_emb.shape)))
        Z = torch.mean(feature_emb, dim=-1, out=None)  # [batch, f, embedding_dim] -> [batch, f]
        A = self.excitation(Z)  # Z:[batch, f] -> 中间变量:[batch, reduced_size] -> A:[batch, f]
        V = feature_emb * A.unsqueeze(-1)  # feature_emb:[batch, f, embedding_dim] A.unsqueeze(-1):[batch, f, 1]
        return V, A


class BilinearInteractionLayer(nn.Module):

    def __init__(self, num_fields, embedding_dim, bilinear_type="field_interaction"):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False) for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False) for i, j in combinations(range(num_fields), 2)])

    def forward(self, feature_emb):
        # feature_emb : [batch, num_fileds, embedding_dim]
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j] for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1] for i, v in enumerate(combinations(feature_emb_list, 2))]
        else:
            raise NotImplementedError
        return torch.cat(bilinear_list, dim=1)


#FiBiNET
class FiBiNET(nn.Module):

    def __init__(
        self,
        embedding_dim=32,
        hidden_units=[64, 64, 64],
        reduction_ratio=3,
        bilinear_type='field_interaction',
        loss_fun='bce',
        enc_dict=None,
    ):
        super(FiBiNET, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.reduction_ratio = reduction_ratio
        self.bilinear_type = bilinear_type
        self.enc_dict = enc_dict
        self.loss_func = set_loss_func(loss_fun)

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        num_sparse, num_dense = get_feature_num(self.enc_dict)

        self.senet = SENETLayer(num_fields=num_sparse)
        self.bilinear = BilinearInteractionLayer(num_fields=num_sparse, embedding_dim=self.embedding_dim, bilinear_type=self.bilinear_type)
        self.lr = LogRegLayer(enc_dict=enc_dict)  #一阶

        # (n-1)*n /2 * embedding_dim + (n-1)*n /2 * embedding_dim + num_dense
        self.dnn_input_dim = num_sparse * (num_sparse - 1) * self.embedding_dim + num_dense
        #sparse_num * emb_dim + dense_num

        self.dnn = MLPLayer(
            input_dim=self.dnn_input_dim,
            output_dim=1,
            hidden_units=self.hidden_units,
            hidden_activation='relu',
            dropout_rate=0,
            final_activation=None,
        )

    def forward(self, data):
        sparse_embedding = self.embedding_layer(data)
        sparse_embedding = torch.stack(sparse_embedding, 1).squeeze(2)  #[batch,num_sparse,embedding_dim]
        dense_input = get_linear_input(self.enc_dict, data)

        lr_logit = self.lr(data)

        # SENET
        senet_embedding, _ = self.senet(sparse_embedding)

        #Bilinear-Interaction
        p = self.bilinear(sparse_embedding)  #[batch, (n-1)*n/2 ,embedding_dim]
        q = self.bilinear(senet_embedding)  #[batch, (n-1)*n/2 ,embedding_dim]

        # Combination Layer
        c = torch.flatten(torch.cat([p, q], dim=1), start_dim=1)  #[batch, (n-1)*n/2 ,embedding_dim] -> [batch, (n-1)*n/2 * embedding_dim]
        dnn_input = torch.cat((c, dense_input), dim=1)  # [batch, dnn_input_dim]
        # DNN
        dnn_logit = self.dnn(dnn_input)

        #输出
        y_pred = torch.sigmoid(lr_logit + dnn_logit)
        loss = self.loss_func(y_pred, data["label"].view(-1, 1).float())
        output_dict = {'y_pred': y_pred, 'loss': loss}
        return output_dict
