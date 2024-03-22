'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-22
Description  : 
'''

from data.data_config import DATA_CONFIG
from utils.logger import logger


def config_middleware():

    def decorator(func):

        def wrapper(config):
            if config is None:
                raise ValueError("config is None")
            required_params = []
            missing_params = [param for param in required_params if param not in config.keys()]
            # 设置默认值
            default_params = {
                "data": "criteo",
                "train_ratio": 0.7,
                "valid_ratio": 0.2,
                "neg_sample_ratio": 3,
                "debug_mode": True,
                "epoch": 5,
                "batch_size": 1024,
                "lr": 0.0001,
                "device": -1,
            }
            if len(missing_params) > 0:
                raise ValueError(f"缺少必要参数: {missing_params}")
            for param in default_params.keys():
                if param not in config.keys():
                    config[param] = default_params[param]
                    logger.warning(f"缺少可选参数 {param}，设置默认值: {default_params[param]}")
            if config["data"] == "criteo":
                config.update(DATA_CONFIG["criteo"])
            elif config["data"] == "movielens":
                config.update(DATA_CONFIG["movielens"])
            elif config["data"] == "amazon":
                config.update(DATA_CONFIG["amazon"])
            else:
                raise ValueError(f"不支持的数据集: {config['data']}")
            logger.send_message(config, message_type=0, message_content_type=0)
            return func(config)

        return wrapper

    return decorator
