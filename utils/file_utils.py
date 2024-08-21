'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-08-14
Description  : 
'''
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
