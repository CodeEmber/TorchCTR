'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-04-11
Description  : 
'''
from trainers.base_train import BaseTrainer
from trainers.dl_trainer import DeepLearningTrainer
from trainers.gnn_trainer import GraphNeuralNetworkTrainer


class TrainManager():

    def __init__(self, config):
        self.config = config
        self.trainer = self._get_trainer()

    def _get_trainer(self):
        if self.config.get('trainer') == 'dl':
            return DeepLearningTrainer()
        elif self.config.get('trainer') == 'gnn':
            return GraphNeuralNetworkTrainer()
        else:
            return BaseTrainer()
