'''
Author       : wyx-hhhh
Date         : 2023-10-30
LastEditTime : 2024-03-22
Description  : 
'''
import time
from datetime import datetime


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
        return time_obj.strftime("%Y-%m-%d %H:%M:%S")
    else:
        raise TypeError("timestamp的类型应为int")


def get_values_by_keys(data, keys):
    return tuple(data[key] for key in keys)
