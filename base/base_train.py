'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-08-23
Description  : 
'''
from abc import abstractmethod

from managers.evaluation_manager import EvaluationManager


class BaseTrainer():

    def __init__(self, config: dict, evaluation_manager: EvaluationManager):
        self.config = config
        self.evaluation_manager = evaluation_manager

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def test_model(self, *args, **kwargs):
        pass

    def valid_model(self, *args, **kwargs):
        raise NotImplementedError()

    def get_test_predict(self, *args, **kwargs):
        raise NotImplementedError()
