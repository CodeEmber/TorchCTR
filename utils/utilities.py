'''
Author       : wyx-hhhh
Date         : 2023-10-30
LastEditTime : 2024-09-04
Description  : 
'''
import json
import time
from datetime import datetime

import numpy as np
import torch


def get_current_time() -> datetime:
    """获取当前的日期和时间"""
    return datetime.now()


def calculate_time_difference(start_time, end_time):
    """计算两个时间点之间的时间差"""
    diff = end_time - start_time
    return diff


def set_timestamp():
    """为特定事件生成一个时间戳"""
    return int(time.time())


def format_time(timestamp):
    """将时间戳转换为特定格式的时间字符串"""
    if isinstance(timestamp, int):
        time_obj = datetime.fromtimestamp(timestamp)
        return time_obj.strftime("%Y-%m-%d-%H-%M")
    else:
        raise TypeError("timestamp的类型应为int")


def get_values_by_keys(data, keys):
    return tuple(data[key] for key in keys)


def is_serializable(obj):
    """检查对象是否可序列化"""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def clean_data(data):
    """递归清理数据，去除不可序列化的对象"""
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items() if is_serializable(v)}
    elif isinstance(data, list):
        return [clean_data(item) for item in data if is_serializable(item)]
    elif is_serializable(data):
        return data
    else:
        return None  # 或者返回其他默认值


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


import os
from typing import List


def get_file_path(path: List[str] = [], add_sep_before=False, add_sep_affter=False) -> str:
    """获取文件路径

    Args:
        path (List[str], optional): 项目路径+文件路径. Defaults to [].
        add_sep_before (bool, optional): 是否在开头添加分隔符. Defaults to False.
        add_sep_affter (bool, optional): 是否在结尾添加分隔符. Defaults to False.

    Returns:
        str: 返回文件路径
    """
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.sep.join(path)
    all_path = os.path.join(root_path, file_path)
    if add_sep_before:
        all_path = os.sep + all_path
    if add_sep_affter:
        all_path = all_path + os.sep
    return all_path


def get_new_file_path(path: List[str] = [], add_sep_before=False, add_sep_affter=False) -> str:
    """获取文件夹下最新的文件路径

    Args:
        path (List[str], optional): 项目路径+文件路径. Defaults to [].
        add_sep_before (bool, optional): 是否在开头添加分隔符. Defaults to False.
        add_sep_affter (bool, optional): 是否在结尾添加分隔符. Defaults to False.

    Returns:
        str: _description_
    """
    file_path = get_file_path(path, add_sep_before, add_sep_affter)
    file_list = os.listdir(file_path)
    try:
        file_list.sort(key=lambda fn: int(fn.split('_')[-1].split('.')[0]))
    except:
        file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
    file_new = os.path.join(file_path, file_list[-1])
    return file_new


def check_path(path: str):
    """检查路径是否存在，如果不存在则创建文件夹或文件

    Args:
        path (str): 文件或文件夹路径
    """
    if os.path.isdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    elif os.path.isfile(path):
        if not os.path.exists(path):
            with open(path, 'w') as f:
                pass
    else:
        os.makedirs(path)
