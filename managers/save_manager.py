import json
import os
import shutil
from typing import List
import torch
from utils.file_utils import check_folder, get_file_path
from torch.utils.tensorboard.writer import SummaryWriter

from utils.utilities import set_timestamp


class SaveManager():

    def __init__(self, config: dict, logger):
        self.config = config
        self.model_name = config['model_name']
        self.data_name = config['data']
        self.logger = logger

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
        file_path = get_file_path(['results', self.model_name, f'evaluation_{self.data_name}.json'])
        try:
            with open(file_path, 'r+') as f:
                data = json.load(f)
                data.append(metric)
                f.seek(0)
                json.dump(data, f)
        except FileNotFoundError:
            with open(file_path, 'x') as f:
                json.dump([metric], f)

    def save_model(self, model, epoch):
        if not os.path.exists((folder_path := get_file_path(['results', self.model_name, 'save_model']))):
            os.makedirs(folder_path)
        if epoch % 5 == 0 or epoch == self.config['epoch'] - 1:
            file_path = get_file_path(['results', self.model_name, 'save_model', f'{self.model_name}_{self.data_name}_{set_timestamp()}.pth'])
            torch.save(model.state_dict(), file_path)

    def save_tensorboardx(self, epoch: int, train_metric: dict, valid_metric: dict):

        log_path = get_file_path(['results', self.model_name, 'save_tensorboard'])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_path)

        for metric_name, metric_value in train_metric.items():
            writer.add_scalar(f"{self.data_name}_train/" + metric_name, metric_value, epoch)

        if valid_metric:
            for metric_name, metric_value in valid_metric.items():
                writer.add_scalar(f"{self.data_name}_valid/" + metric_name, metric_value, epoch)

    def save_all(
        self,
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
            file_path = get_file_path(['results', self.model_name])
            if os.path.exists(file_path):
                shutil.rmtree(file_path)
            self.logger.info(f"清空{self.model_name}的结果")
        check_folder(get_file_path(['results', self.model_name]))
        if is_save_model:
            self.save_model(model=model, epoch=epoch)
        if is_save_tensorboard:
            self.save_tensorboardx(
                epoch=epoch + 1,
                train_metric=train_metric,
                valid_metric=valid_metric,
            )
        if is_save_evaluation:
            self.save_evaluation_results(metric=[
                {
                    'train': train_metric,
                    'valid': valid_metric,
                    'test': test_metric,
                    "epoch": epoch + 1,
                    "time": set_timestamp()
                },
                other_metric,
            ], )
