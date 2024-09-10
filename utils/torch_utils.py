'''
Author       : wyx-hhhh
Date         : 2024-05-28
LastEditTime : 2024-09-04
Description  : 
'''
'''
Author       : wyx-hhhh
Date         : 2024-05-28
LastEditTime : 2024-09-04
Description  : 
'''
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
