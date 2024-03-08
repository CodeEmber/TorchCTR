'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-05
Description  : 
'''
from typing import List
import torch.nn as nn

from utils.torch_utils import set_activation


class MLPLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        hidden_activation: str | List[str],
        dropout_rate: float | List[float],
        output_dim: int = 0,
        batch_norm: bool = False,
        use_bias: bool = True,
        final_activation: str = None,
    ):
        """MLP层

        Args:
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            hidden_units (List[int]): 隐藏层维度
            hidden_activation (str | List[str]): 隐藏层激活函数
            final_activation (str): 输出层激活函数
            dropout (float | List[float]): 隐藏层dropout
            batch_norm (bool, optional): 是否使用batch_norm. Defaults to False.
            use_bias (bool, optional): 是否使用bias. Defaults to True.
        """
        super(MLPLayer, self).__init__()
        dense_layers = []
        if isinstance(hidden_activation, str):
            hidden_activation = [hidden_activation] * len(hidden_units)
        if isinstance(dropout_rate, float) or isinstance(dropout_rate, int):
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
