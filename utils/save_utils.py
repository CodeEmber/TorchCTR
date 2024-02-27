'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-02-27
Description  : 
'''
import json
import os
import shutil
from typing import List
import torch
from utils.file_utils import check_folder, get_file_path
from torch.utils.tensorboard.writer import SummaryWriter

from utils.time_utils import set_timestamp
from utils.logger import MyLogger

logger = MyLogger()


def save_evaluation_results(model_name: str, metric: List[dict], data_name: str):
    """保存模型评估结果

    Args:
        model_name (str): 模型名称，需要与run_expid.py中的model_name一致
        metric (list[dict] | dict): 存储的评估结果，可以是一个字典，也可以是一个字典列表
    """
    if isinstance(metric, List):
        temp_metric = {}
        for m in metric:
            if m:
                for k, v in m.items():
                    if v is not None:
                        temp_metric[k] = v
        metric = temp_metric
    else:
        raise TypeError("metric的格式为List[dict]")
    file_path = get_file_path(['results', model_name, f'evaluation_{data_name}.json'])
    try:
        with open(file_path, 'r+') as f:
            data = json.load(f)
            data.append(metric)
            f.seek(0)
            json.dump(data, f)
    except FileNotFoundError:
        with open(file_path, 'x') as f:
            json.dump([metric], f)


def save_model(model_name: str, model, data_name: str):
    """保存模型

    Args:
        model_name (str): 模型名称，需要与run_expid.py中的model_name一致
        model ([type]): 模型
    """
    if not os.path.exists((folder_path := get_file_path(['results', model_name, 'save_model']))):
        os.makedirs(folder_path)
    file_path = get_file_path(['results', model_name, 'save_model', f'{model_name}_{data_name}.pth'])
    torch.save(model.state_dict(), file_path)


def save_tensorboardx(model_name: str, epoch: int, train_metric: dict, valid_metric: dict, data_name: str):
    """保存tensorboardx

    Args:
        model_name (str): 模型名称，需要与run_expid.py中的model_name一致
        epoch (int): 当前epoch
        train_metric (dict): 训练集评估结果
        valid_metric (dict): 验证集评估结果
    """
    log_path = get_file_path(['results', model_name, 'save_tensorboard'])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    for metric_name, metric_value in train_metric.items():
        writer.add_scalar(f"{data_name}_train/" + metric_name, metric_value, epoch)

    for metric_name, metric_value in valid_metric.items():
        writer.add_scalar(f"{data_name}_valid/" + metric_name, metric_value, epoch)


def save_all(
    model_name: str,
    data_name: str = None,
    epoch: int = -1,
    is_save_model: bool = True,
    is_save_tensorboard: bool = True,
    is_save_evaluation: bool = True,
    is_clear: bool = False,
    model: torch.nn.Module = None,
    train_metric: dict = None,
    valid_metric: dict = None,
    test_metric: dict = None,
    other_metric: dict = None,
):
    """保存所有结果

    Args:
        model_name (str): 模型名称，需要与run_expid.py中的model_name一致
        data_name (str): 数据名称
        is_save_model (bool): 是否保存模型
        is_save_tensorboard (bool): 是否保存tensorboardx
        is_save_evaluation (bool): 是否保存评估结果
        is_clear (bool): 是否清空之前的结果
        model (torch.nn.Module): 模型
        train_metirc (dict): 训练集评估结果
        valid_metric (dict): 验证集评估结果
        test_metric (dict): 测试集评估结果
        other_metric (dict): 其他评估结果
    """
    if is_clear and epoch == 0:
        file_path = get_file_path(['results', model_name])
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        logger.info(f"清空{model_name}的结果")
    check_folder(get_file_path(['results', model_name]))
    if is_save_model:
        save_model(
            model_name=model_name,
            model=model,
            data_name=data_name,
        )
    if is_save_tensorboard:
        save_tensorboardx(
            model_name=model_name,
            data_name=data_name,
            epoch=epoch + 1,
            train_metric=train_metric,
            valid_metric=valid_metric,
        )
    if is_save_evaluation:
        save_evaluation_results(
            model_name=model_name,
            data_name=data_name,
            metric=[
                {
                    'train': train_metric,
                    'valid': valid_metric,
                    'test': test_metric,
                    "epoch": epoch + 1,
                    "time": set_timestamp()
                },
                other_metric,
            ],
        )
