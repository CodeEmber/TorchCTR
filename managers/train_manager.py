'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-03-22
Description  : 
'''
from trainers.amazon_train import AmazonTrainer
from trainers.base_train import BaseTrainer
from trainers.criteo_train import CriteoTrainer
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
        else:
            return BaseTrainer()

    def train_model(self, model, train_loader, optimizer, device, metric_func=["roc_auc_score", "log_loss"]):
        return self.trainer.train_model(model, train_loader, optimizer, device, metric_func)

    def valid_model(self, model, valid_loader, device, metric_func=["roc_auc_score", "log_loss"]):
        return self.trainer.valid_model(model, valid_loader, device, metric_func)

    def test_model(self, model, test_loader, device, metric_func=["roc_auc_score", "log_loss"]):
        return self.trainer.test_model(model, test_loader, device, metric_func)

    def get_test_predict(self, model, data_loader, device):
        return self.trainer.get_test_predict(model, data_loader, device)
