'''
Author       : wyx-hhhh
Date         : 2024-03-22
LastEditTime : 2024-09-04
Description  : 
'''
# from trainers.base_train import BaseTrainer
# from trainers.dl_trainer import DeepLearningTrainer
# from trainers.gnn_trainer import GnnTrainer

# class TrainManager():

#     def __init__(self, config, evaluation_manager=None):
#         self.config = config
#         self.evaluation_manager = evaluation_manager
#         self.trainer = self._get_trainer()

#     def _get_trainer(self):
#         if self.config.get('trainer') == 'dl':
#             return DeepLearningTrainer(self.config, self.evaluation_manager)
#         elif self.config.get('trainer') == 'gnn':
#             return GnnTrainer(self.config, self.evaluation_manager)
#         else:
#             return BaseTrainer()
