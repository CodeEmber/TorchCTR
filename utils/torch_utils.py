from ast import Raise
from typing import Optional
import torch
import torch.nn as nn


def set_device(device: int) -> torch.device:
    """设置设备

    Args:
        device (int): 设备号

    Returns:
        torch.device: 设备
    """
    if device == -1:
        return torch.device("cpu")
    elif device == -2:
        return torch.device("mps")
    else:
        return torch.device(f"cuda:{device}")


def set_activation(activation: str) -> nn.Module:
    """返回激活函数

    Args:
        activation (str): 激活函数名

    Raises:
        ValueError: 只允许relu, sigmoid, tanh, softmax

    Returns:
        Optional[nn.Module]: 返回激活函数
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softmax":
        return nn.Softmax()
    else:
        raise ValueError("activation must be relu, sigmoid, tanh or softmax")


def set_loss_func(loss_func: str) -> nn.Module:
    """返回损失函数

    Args:
        loss_func (str): 损失函数名

    Raises:
        ValueError: 只允许mse, bce, ce

    Returns:
        Optional[nn.Module]: 返回损失函数
    """
    if loss_func == "mse":
        return nn.MSELoss()
    elif loss_func == "bce":
        return nn.BCELoss()
    elif loss_func == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("loss_func must be mse, bce or ce")


def get_feature_num(enc_dict):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense += 1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse += 1
    return num_sparse, num_dense


def get_dnn_input_dim(enc_dict: dict, embedding_dim: int) -> int:
    """返回dnn的输入维度

    Args:
        enc_dict (dict): 编码后的特征字典
        embedding_dim (int): embedding维度

    Returns:
        int: dnn的输入维度
    """
    num_sparse, num_dense = get_feature_num(enc_dict)
    return num_sparse * embedding_dim + num_dense


def get_linear_input(enc_dict: dict, data: dict) -> torch.Tensor:
    """获取线性部分的输入

    Args:
        enc_dict (dict): 编码后的特征字典
        data (dict): 数据

    Returns:
        torch.Tensor: 线性部分的输入
    """
    res_data = []
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            res_data.append(data[col])
    res_data = torch.stack(res_data, axis=1)
    return res_data
