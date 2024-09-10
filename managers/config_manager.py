'''
Author       : wyx-hhhh
Date         : 2024-03-24
LastEditTime : 2024-08-26
Description  : 
'''
from config.data_config import DATA_CONFIG
from config.global_config import GOLBAL_CONFIG
from config.save_config import SAVE_CONFIG
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
                "valid": [
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

    def _get_train_config(self):
        train_config = self.train_config.copy()
        default_config = train_config.pop("default", {})
        data_specific_config = train_config.pop("data_specific_config", {}).get(train_config.get("data"), {})

        # 更新配置
        train_config.update(default_config)
        train_config.update(data_specific_config)

        self.train_config = train_config

    def _get_data_config(self):
        self.data_config = DATA_CONFIG.get(self.train_config.get('data'))
        if self.data_config is None:
            raise ValueError(f"不支持的数据集: {self.train_config['data']}")

    def _get_global_config(self):
        self.global_config = GOLBAL_CONFIG
        # debug模式下不发送slack消息
        if self.train_config.get('debug_mode'):
            self.global_config['is_slack_enabled'] = False

    def _get_save_config(self):
        self.save_config = SAVE_CONFIG

    def _check_params(self):
        missing_params = [param for param in self.required_params if param not in self.train_config.keys()]
        if len(missing_params) > 0:
            raise ValueError(f"缺少必要参数: {missing_params}")
        for param in self.default_train_config.keys():
            if param not in self.train_config.keys():
                self.train_config[param] = self.default_train_config[param]
                self.logger.warning(f"缺少可选参数 {param}，设置默认值: {self.default_train_config[param]}")

    def _log_params(self, config):
        if config["debug_mode"]:
            self.logger.info("注意当前处于debug模式，数据集较小，仅用于调试")
        else:
            self.logger.info("注意当前不处于debug模式，数据集较大，用于正式训练")
        if not config["debug_mode"] and config["epoch"] < 10:
            self.logger.warning("非debug模式下，建议epoch至少设置为10")
        self.logger.info("\n")
        self.logger.info(f"Global Hyper Parameters:")
        for key, value in self.global_config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("\n")

        self.logger.info(f"Data Hyper Parameters:")
        for key, value in self.data_config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("\n")

        self.logger.info(f"Training Hyper Parameters:")
        for key, value in self.train_config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("\n")

        self.logger.info(f"Save Hyper Parameters:")
        for key, value in self.save_config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("\n")

    def get_config(self):
        self._get_train_config()
        self._get_data_config()
        self._get_global_config()
        self._get_save_config()
        self.logger = LoggerManager(config=self.global_config)
        self._check_params()
        all_config = {**self.global_config, **self.data_config, **self.save_config, **self.train_config}
        if all_config.get("is_auto_gpu", True):
            GPUMonitor(config=self.train_config, logger=self.logger)
            all_config.update({"device": self.train_config["device"]})
        self._log_params(all_config)
        return all_config
