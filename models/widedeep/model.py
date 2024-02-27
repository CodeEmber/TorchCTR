from typing import List
import torch
import torch.nn as nn
from layers import EmbeddingLayer
from utils.torch_utils import get_dnn_input_dim, get_linear_input, set_activation, set_loss_func


class Weed(nn.Module):

    def __init__(self, enc_dict: dict = None):
        super(Weed, self).__init__()
        self.enc_dict = enc_dict
        self.emb_layer = EmbeddingLayer(self.enc_dict, embedding_dim=1)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, embedding_dim=1)
        self.fc = nn.Linear(self.dnn_input_dim, 1)

    def forward(self, x):
        sparse_emb_list = self.emb_layer(x)
        sparse_emb = torch.stack(sparse_emb_list, dim=1).flatten(start_dim=1)
        dense_input = get_linear_input(self.enc_dict, x)
        dnn_input = torch.cat((sparse_emb, dense_input), dim=1)
        out = self.fc(dnn_input)
        return out


class Deep(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: List[int],
        hidden_activation: str | List[str],
        final_activation: str,
        dropout_rate: float | List[float],
        batch_norm: bool = False,
        use_bias: bool = True,
    ):
        """初始化Deep

        Args:
            input_dim (int): 输入层维度
            output_dim (int): 输出层维度，二分类一般为1，多分类为类别数
            hidden_units (List[int]): 隐藏层维度
            hidden_activation (str|List[str]): 隐藏层激活函数
            final_activation (str): 输出层激活函数
            dropout_rate (float|List[str]): 隐藏层dropout比例
            batch_norm (bool, optional): 是否在隐藏层后进行归一化操作. Defaults to False.
            use_bias (bool, optional): 是否在全连接层使用偏置项. Defaults to True.
        """
        super(Deep, self).__init__()
        dense_layers = []
        if isinstance(hidden_activation, str):
            hidden_activation = [hidden_activation] * len(hidden_units)
        if isinstance(dropout_rate, float):
            dropout_rate = [dropout_rate] * len(hidden_units)
        hidden_activation = [set_activation(act) for act in hidden_activation]
        hidden_units = [input_dim] + hidden_units
        for index in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[index], hidden_units[index + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[index + 1]))
            if hidden_activation[index]:
                dense_layers.append(hidden_activation[index])
            if dropout_rate[index] > 0:
                dense_layers.append(nn.Dropout(dropout_rate[index]))
        if output_dim > 0:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if final_activation:
            dense_layers.append(set_activation(final_activation))
        self.dnn = nn.Sequential(*dense_layers)

    def forward(self, x):
        return self.dnn(x)


class WDL(nn.Module):

    def __init__(
        self,
        enc_dict: dict = None,
        embedding_dim: int = 40,
        hidden_units: List[int] = [64, 64, 64],
        hidden_activation: str | List[str] = "relu",
        final_activation: str = "sigmoid",
        dropout_rate: float | List[str] = 0.5,
        batch_norm: bool = False,
        use_bias: bool = True,
        loss_func: str = "bce",
    ):
        """初始化Wide&Deep
    
            Args:
                enc_dict (dict, optional): 编码后的特征字典. Defaults to None.
                hidden_units (List[int], optional): 隐藏层维度. Defaults to None.
                hidden_activation (str|List[str], optional): 隐藏层激活函数. Defaults to "relu".
                final_activation (str, optional): 输出层激活函数. Defaults to "sigmoid".
                dropout_rate (float|List[str], optional): 隐藏层dropout比例. Defaults to 0.5.
                batch_norm (bool, optional): 是否在隐藏层后进行归一化操作. Defaults to False.
                use_bias (bool, optional): 是否在全连接层使用偏置项. Defaults to True.
            """
        super(WDL, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.loss_func = set_loss_func(loss_func)
        self.emb_layer = EmbeddingLayer(self.enc_dict, embedding_dim=self.embedding_dim)
        # weed部分
        self.wide = Weed(self.enc_dict)
        # deep部分
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, embedding_dim=self.embedding_dim)
        self.dnn = Deep(
            input_dim=self.dnn_input_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            final_activation=final_activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            use_bias=use_bias,
        )

    def forward(self, x):
        wide_out = self.wide(x)

        sparse_emb_list = self.emb_layer(x)
        sparse_emb = torch.stack(sparse_emb_list, dim=1).flatten(start_dim=1)

        dense_input = get_linear_input(self.enc_dict, x)
        dnn_input = torch.cat([sparse_emb, dense_input], dim=1)
        deep = self.dnn(dnn_input)

        y_pred = torch.sigmoid(wide_out + deep)
        loss = self.loss_func(y_pred, x["label"].float().view(-1, 1))
        output_dict = {"y_pred": y_pred, "loss": loss}
        return output_dict
