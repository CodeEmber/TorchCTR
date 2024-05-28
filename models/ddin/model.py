from itertools import combinations
from typing import List
import torch
import torch.nn as nn
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
        history_seq_list = []
        for col in self.item_cols:
            item_emb_list.append(self.embedding_dict[col](data[col]))
        item_emb = torch.cat(item_emb_list, dim=1).squeeze(1)
        for col in self.user_cols:
            user_emb_list.append(self.embedding_dict[col](data[col]))
        user_emb = torch.cat(user_emb_list, dim=1)

        for col in self.history_cols:
            history_seq_list.append(self.embedding_dict[self.enc_dict[col]['share_with']](data[col]))
        history_seq = torch.cat(history_seq_list, dim=-1)
        return user_emb, item_emb, history_seq


class Dice(nn.Module):

    def __init__(self, input_dim: int, alpha: float = 0., eps=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float))

    def forward(self, x):
        px = torch.sigmoid(self.bn(x))
        output = self.alpha * (1 - px) * x + px * x  # (batch_size, input_dim)
        return output


class DIEBLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_unit: int = 32,
    ):
        super(DIEBLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_unit = hidden_unit

        self.lr_wkp = nn.Linear(self.embedding_dim * 2, self.hidden_unit)
        self.lr_wq = nn.Linear(self.embedding_dim * 2, self.hidden_unit)
        self.lr_wku = nn.Linear(self.embedding_dim * 2, self.hidden_unit)

        self.dnn1 = nn.Sequential(nn.ReLU(), nn.Linear(self.hidden_unit * 3, 1), nn.Softmax(dim=-1))

        self.dnn2 = nn.Sequential(nn.ReLU(), nn.Linear(self.hidden_unit * 3, 1), nn.Softmax(dim=-1))

    def get_mask(self, history_seq):
        return (history_seq != 0).sum(dim=-1)

    def forward(self, item_emb, history_seq):
        # item_emb: (batch_size, embedding_dim)  (1024，32)
        # history_seq: (batch_size, history_len, embedding_dim)  (1024,20,32)
        mask = self.get_mask(history_seq)  # (batch_size, history_len) (1024,20)
        efficient_history_seq_len = mask.count_nonzero(dim=-1)  # (batch_size,) 有效的历史行为序列长度 (1024,)
        mean_history_seq = history_seq.sum(dim=1) / efficient_history_seq_len.unsqueeze(-1)  # (batch_size, embedding_dim) (1024,32)

        history_seq_kp = self.lr_wkp(history_seq)  # (batch_size, history_len, hidden_unit) (1024,20,32)
        item_emb_q = self.lr_wq(item_emb).unsqueeze(1).expand(-1, history_seq.size(1), -1)  # (batch_size, history_len, hidden_unit) (1024,20,32)
        mean_history_seq = mean_history_seq.unsqueeze(1).expand(-1, history_seq.size(1), -1)  # (batch_size, history_len, embedding_dim) (1024,20,32)
        mean_history_seq_mp = self.lr_wkp(mean_history_seq) / efficient_history_seq_len.unsqueeze(1).unsqueeze(2).expand(-1, 20, -1)  # (batch_size, history_len, hidden_unit) (1024,20,32)
        mean_history_seq_mu = self.lr_wku(mean_history_seq) / efficient_history_seq_len.unsqueeze(1).unsqueeze(2).expand(-1, 20, -1)  # (batch_size, history_len, hidden_unit) (1024,20,32)

        history_item_whiten_mean_history_seq_mp = (history_seq_kp - mean_history_seq_mp)  # (batch_size, history_len, hidden_unit) (1024,20,32)
        history_item_dot_mean_history_seq_mu = (history_item_whiten_mean_history_seq_mp * mean_history_seq_mu)  # (batch_size, history_len, hidden_unit) (1024,20,32)

        dnn1_input = torch.cat([history_item_whiten_mean_history_seq_mp, history_item_dot_mean_history_seq_mu, item_emb_q], dim=-1)  # (batch_size, history_len, hidden_unit * 3) (1024,20,96)
        dnn2_input = torch.cat([history_item_whiten_mean_history_seq_mp, history_item_dot_mean_history_seq_mu, mean_history_seq_mu], dim=-1)  # (batch_size, history_len, hidden_unit * 3) (1024,20,96)

        dnn1_input = dnn1_input.view(-1, self.hidden_unit * 3)  # (batch_size * history_len, hidden_unit * 3) (20480,96)
        dnn2_input = dnn2_input.view(-1, self.hidden_unit * 3)  # (batch_size * history_len, hidden_unit * 3) (20480,96)

        dnn1_output = self.dnn1(dnn1_input)  # (batch_size * history_len, 1) (20480,1)
        dnn2_output = self.dnn2(dnn2_input)  # (batch_size * history_len, 1) (20480,1)

        dnn1_output = dnn1_output.view(-1, history_seq.size(1))  # (batch_size, history_len) (1024,20)
        dnn1_output = dnn1_output.unsqueeze(-1) * history_seq_kp  # (batch_size, history_len, hidden_unit) (1024,20,32)

        dnn2_output = dnn2_output.view(-1, history_seq.size(1))  # (batch_size, history_len) (1024,20)
        dnn2_output = dnn2_output.unsqueeze(-1) * mean_history_seq_mu  # (batch_size, history_len, hidden_unit) (1024,20,32)

        output = (dnn1_output + dnn2_output).sum(dim=1)  # (batch_size, hidden_unit) (1024,32)
        return output


class DeepLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_units: List[int],
        item_label_num: int,
        user_label_num: int,
    ):
        super(DeepLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.item_label_num = item_label_num
        self.user_label_num = user_label_num
        self.mlp_layer = MLPLayer(
            input_dim=self.embedding_dim * (2 * self.item_label_num + self.user_label_num),
            hidden_units=self.hidden_units,
            output_dim=1,
        )

    def forward(self, x):
        out = self.mlp_layer(x)
        out = torch.sigmoid(out)
        return out


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
        feature_emb_list = [feature_emb_list[:, i * 16:(i + 1) * 16].view(1024, 1, 16) for i in range(5)]
        pair_feature = []
        for vi, vj in combinations(feature_emb_list, 2):
            pair_feature.append(vi * vj)
        pair_feature = torch.cat(tensors=pair_feature, dim=1)  # [batch,num_pair:<(n-1)*n/2>,emb]
        attention_weight = self.attention_net(pair_feature)  # [batch,num_pair,1]
        if self.dropout_rate > 0:
            attention_weight = self.dropout(attention_weight)
        pair_feature = torch.sum(pair_feature * attention_weight, dim=1)  # [batch,emb]
        pair_feature = torch.sum(pair_feature, dim=-1).unsqueeze(-1)  # [batch,1]
        return pair_feature


class AFMLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        item_label_num: int,
        user_label_num: int,
        attention_factor: int = 32,
        dropout_rate: int = 0,
        l2_reg: float = 0.1,
        loss_fun: str = 'bce',
    ):
        super(AFMLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.item_label_num = item_label_num
        self.user_label_num = user_label_num
        self.attention_factor = attention_factor
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.loss_fun = set_loss_func(loss_fun)
        input_dim = self.embedding_dim * (2 * self.item_label_num + self.user_label_num)
        self.fm = FMPlus(
            embedding_dim=self.embedding_dim,
            attention_factor=attention_factor,
            dropout_rate=dropout_rate,
        )  #二阶
        self.lr = nn.Linear(input_dim, 1)  #一阶

    def get_l2_reg_loss(self, ):
        attention_W = self.fm.attention_net[0].weight
        reg_loss = torch.sum(self.l2_reg * attention_W**2)
        return reg_loss

    def forward(self, data):
        # FM
        lr_logit = self.lr(data)  #一阶交叉
        fm_logit = self.fm(data)  #二阶交叉

        #输出
        out = torch.sigmoid(lr_logit + fm_logit)
        return out


class DDIN(nn.Module):

    def __init__(
        self,
        enc_dict: dict = None,
        hidden_units: List[int] = [64, 32, 16],
        attention_units: List[int] = [32],
        embedding_dim: int = 16,
        loss_func: str = "bce",
    ):
        super(DDIN, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.attention_units = attention_units
        self.loss_func = set_loss_func(loss_func)
        self.count_map = self.enc_dict['count_map']

        self.embedding_layer = EmbeddingLayer(self.enc_dict, self.embedding_dim)
        self.user_label_num, self.item_label_num = self.get_feature_num()
        self.dieb_layer = DIEBLayer()
        self.mlp_layer = DeepLayer(
            embedding_dim=self.embedding_dim,
            hidden_units=self.hidden_units,
            item_label_num=self.item_label_num,
            user_label_num=self.user_label_num,
        )
        self.afm_layer = AFMLayer(
            embedding_dim=self.embedding_dim,
            item_label_num=self.item_label_num,
            user_label_num=self.user_label_num,
            attention_factor=self.attention_units[0],
            dropout_rate=0,
            l2_reg=0.1,
            loss_fun='bce',
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
        tag_mba_l2 = self.MBA_Reg(data, 'item_target_tag')
        return user_mba_l2 + item_mba_l2 + tag_mba_l2

    def forward(self, data):
        # (batch_size, embedding_dim*user_label_num)，(batch_size, embedding_dim*item_label_num)，(batch_size, history_len, embedding_dim*item_label_num)
        # ([1024,16])  ([1024,32])  ([1024,20,32])
        user_emb, item_emb, history_seq = self.embedding_layer(data)
        attention_output = self.dieb_layer(item_emb, history_seq)
        cat_input = torch.cat([user_emb, item_emb, attention_output], dim=1)
        mlp_output = self.mlp_layer(cat_input)
        afm_output = self.afm_layer(cat_input)
        y_pred = mlp_output + afm_output
        y_pred = torch.sigmoid(y_pred)
        loss = self.loss_func(y_pred, data["label"].float().view(-1, 1)) + 0.2 * self.lr_reg(data)
        output_dict = {"y_pred": y_pred, "loss": loss}
        return output_dict
