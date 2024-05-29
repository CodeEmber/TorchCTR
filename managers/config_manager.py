'''
Author       : wyx-hhhh
Date         : 2024-03-24
LastEditTime : 2024-05-29
Description  : 
'''
from config.data_config import DATA_CONFIG
from config.global_config import GOLBAL_CONFIG
from managers.logger_manager import LoggerManager
from utils.gpu_utils import GPUMonitor


class ConfigManager():

    def __init__(self, train_config):
        self.train_config = train_config
        self.data_config = {}
        self.default_train_config = {
            "debug_mode": True,
            "epoch": 5,
            "batch_size": 1024,
            "lr": 0.0001,
            "device": -1,
            "metric_func": {
                "train": [
                    {
                        "eval_func": "auc"
                    },
                    {
                        "eval_func": "log_loss"
                    },
                ],
                "eval": [
                    {
                        "eval_func": "ndcg",
                        "k": [10, 20, 50]
                    },
                    {
                        "eval_func": "gauc"
                    },
                ],
            },
        }
        self.global_config = {}
        self.required_params = ["data", "model_name", "trainer"]

    def _get_data_config(self):
        self.data_config = DATA_CONFIG.get(self.train_config.get('data'))
        if self.data_config is None:
            raise ValueError(f"不支持的数据集: {self.train_config['data']}")

    def _get_global_config(self):
        self.global_config = GOLBAL_CONFIG

    def _check_params(self):
        missing_params = [param for param in self.required_params if param not in self.train_config.keys()]
        if len(missing_params) > 0:
            raise ValueError(f"缺少必要参数: {missing_params}")
        for param in self.default_train_config.keys():
            if param not in self.train_config.keys():
                self.train_config[param] = self.default_train_config[param]
                self.logger.warning(f"缺少可选参数 {param}，设置默认值: {self.default_train_config[param]}")

    def get_config(self):
        self._get_data_config()
        self._get_global_config()
        self.logger = LoggerManager(config=self.global_config)
        GPUMonitor(config=self.train_config, logger=self.logger)
        self._check_params()
        all_config = {**self.global_config, **self.data_config, **self.train_config}
        self.logger.send_message({**self.data_config, **self.train_config}, message_type=0, message_content_type=0)
        return all_config
