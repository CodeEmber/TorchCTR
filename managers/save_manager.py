import json
import os
import shutil
from typing import List

import numpy as np
import pandas as pd
import torch
from utils.file_utils import check_path, get_file_path
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

from utils.utilities import format_time, get_current_time, set_timestamp


class EarlyStopping:

    def __init__(self, patience=5, verbose=False, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # 'min' 或 'max'
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()  # 保存最佳模型
        elif (self.mode == 'min' and score < self.best_score) or (self.mode == 'max' and score > self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_model = model.state_dict()  # 保存最佳模型
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # 触发早停


class SaveManager():

    def __init__(self, config: dict, logger):
        self.config = config
        self.model_name = config['model_name']
        self.data_name = config['data']
        self.logger = logger
        self._get_save_path()
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            mode=config['early_stopping_mode'],
            verbose=config["is_verbose"],
        )

    def _get_save_path(self):
        time_str = format_time(set_timestamp())
        self.root_path = get_file_path(['results', self.model_name])
        self.model_save_path = get_file_path([self.root_path, "save_model", f"{self.model_name}_{self.data_name}_{time_str}.pth"])
        self.tensorboardx_save_path = get_file_path([self.root_path, "save_tensorboard", f"{self.model_name}_{self.data_name}_{time_str}"])
        self.evaluation_save_path = get_file_path([self.root_path, "evaluation", f"{self.model_name}_{self.data_name}_{time_str}.json"])
        check_path(self.root_path)
        check_path(get_file_path([self.root_path, "save_model"]))
        check_path(get_file_path([self.root_path, "save_tensorboard"]))
        check_path(get_file_path([self.root_path, "evaluation"]))
        save_path_dict = {
            "model_save_path": self.model_save_path,
            "tensorboardx_save_path": self.tensorboardx_save_path,
            "evaluation_save_path": self.evaluation_save_path,
        }
        self.logger.info("Save Path:")
        for k, v in save_path_dict.items():
            self.logger.info(f"{k}: {v}")

    def save_evaluation_results(self, metric: List[dict]):
        if isinstance(metric, List):
            temp_metric = {}
            for m in metric:
                if m:
                    for k, v in m.items():
                        if v is not None:
                            temp_metric[k] = v
            metric = temp_metric
            metric.update({"model_name": self.model_name, "data_name": self.data_name})
            self.logger.send_message(metric, message_type=0, message_content_type=0)
        else:
            raise TypeError("metric的格式为List[dict]")
        try:
            with open(self.evaluation_save_path, 'r+') as f:
                data = json.load(f)
                data.append(metric)
                f.seek(0)
                json.dump(data, f)
        except FileNotFoundError:
            with open(self.evaluation_save_path, 'x') as f:
                json.dump([metric], f)

    def save_model(self, model):
        if not os.path.exists((folder_path := get_file_path([self.root_path, "save_model"]))):
            os.makedirs(folder_path)
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), self.model_save_path)
        else:
            torch.save(model, self.model_save_path)
        # torch.save(model.state_dict(), self.model_save_path)

    def save_tensorboardx(
        self,
        epoch: int,
        train_metric: dict = None,
        valid_metric: dict = None,
        test_metric: dict = None,
        other_metric: dict = None,
    ):
        if not os.path.exists(self.tensorboardx_save_path):
            os.makedirs(self.tensorboardx_save_path)
        writer = SummaryWriter(self.tensorboardx_save_path)

        if train_metric:
            for metric_name, metric_value in train_metric.items():
                writer.add_scalar(f"{self.data_name}_train/" + metric_name, metric_value, epoch)

        if valid_metric:
            for metric_name, metric_value in valid_metric.items():
                writer.add_scalar(f"{self.data_name}_valid/" + metric_name, metric_value, epoch)

        if test_metric:
            for metric_name, metric_value in test_metric.items():
                writer.add_scalar(f"{self.data_name}_test/" + metric_name, metric_value, epoch)

        if other_metric:
            for metric_name, metric_value in other_metric.items():
                writer.add_scalar(f"{self.data_name}_other/" + metric_name, metric_value, epoch)

    def save_csv(self, data: dict, file_path: str):
        check_path(file_path)
        data = pd.DataFrame(data)
        data.to_csv(file_path, index=False)
        self.logger.info(f"保存{file_path}成功")

    def save_json(self, data: dict, file_path: str):
        check_path(file_path)
        with open(file_path, 'w') as f:
            json.dump(data, f)
        self.logger.info(f"保存{file_path}成功")

    def save_all(
        self,
        epoch: int = -1,
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
            model (torch.nn.Module): 模型
            train_metirc (dict): 训练集评估结果
            valid_metric (dict): 验证集评估结果
            test_metric (dict): 测试集评估结果
            other_metric (dict): 其他评估结果
        """
        if test_metric:
            self.early_stopping(test_metric[self.config["early_stopping_score"]], model)
        if valid_metric:
            self.early_stopping(valid_metric[self.config["early_stopping_score"]], model)
        if self.early_stopping.early_stop:
            self.config["early_stop"] = True
        if self.config["is_save_model"]:
            if epoch == 0 or self.early_stopping.early_stop:
                self.save_model(model=self.early_stopping.best_model)
        if self.config["is_save_tensorboard"]:
            self.save_tensorboardx(
                epoch=epoch,
                train_metric=train_metric,
                valid_metric=valid_metric,
                test_metric=test_metric,
                other_metric=other_metric,
            )
        if self.config["is_save_evaluation"]:
            self.save_evaluation_results(metric=[
                {
                    'train': train_metric,
                    'valid': valid_metric,
                    'test': test_metric,
                    "epoch": epoch,
                    "time": set_timestamp()
                },
                other_metric,
            ], )
