'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-04-11
Description  : 
'''
from trainers.amazon_train import AmazonTrainer
from trainers.base_train import BaseTrainer
from trainers.criteo_train import CriteoTrainer
from trainers.gowalla_train import GowallaTrainer
from trainers.movielens_train import MovieLensTrainer


class TrainManager():

    def __init__(self, config):
        self.config = config
        self.trainer = self._get_trainer()

    def _get_trainer(self):
        if self.config.get('data') == 'criteo':
            return CriteoTrainer()
        elif self.config.get('data') == 'movielens':
            return MovieLensTrainer()
        elif self.config.get('data') == 'amazon':
            return AmazonTrainer()
        elif self.config.get('data') == 'gowalla':
            return GowallaTrainer()
        else:
            return BaseTrainer()
