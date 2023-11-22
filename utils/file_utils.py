'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2023-10-30
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


def check_folder(folder_path: str):
    """检查文件夹是否存在，如果不存在则创建

    Args:
        folder_path (str): 文件夹路径
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
