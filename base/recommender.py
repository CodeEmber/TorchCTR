'''
Author       : wyx-hhhh
Date         : 2024-09-04
LastEditTime : 2024-09-05
Description  : 
'''
from abc import abstractmethod
from managers import ConfigManager, DataManager, LoggerManager, SaveManager, EvaluationManager
from utils.utilities import set_seed


class Recommender(object):

    def __init__(self, train_config: dict) -> None:
        config = ConfigManager(train_config=train_config).get_config()
        logger = LoggerManager(config=config)
        config['logger'] = logger
        set_seed(config.get("seed", 2024))
        self.config = config
        self.data_dict = DataManager(config=config).data_process()
        self.evaluation_manager = EvaluationManager(config=config)
        self.save_manager = SaveManager(config=config)
        self.config["logger"].send_message(config, message_type=0, message_content_type=0)
        self.config["data_dict"] = self.data_dict

    @abstractmethod
    def train(self, args):
        pass

    def valid(self):
        pass

    @abstractmethod
    def test(self, args):
        pass

    def predict(self):
        pass

    @abstractmethod
    def run(self):
        pass
