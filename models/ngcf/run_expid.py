'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-08-20
Description  : 
'''
import torch

from models.ngcf.model import NGCF
from models.ngcf.train_config import train_config
from managers import ConfigManager, DataManager, LoggerManager, TrainManager, SaveManager, EvaluationManager
from utils.torch_utils import set_device
from utils.utilities import get_values_by_keys, set_seed
from trainers.gnn_trainer import GraphNeuralNetworkTrainer

config_manager = ConfigManager(train_config=train_config)
config = config_manager.get_config()
set_seed(config.get("seed", 2024))
logger = LoggerManager(config=config)
config['logger'] = logger
data_manager = DataManager(config=config)
data_dict = data_manager.data_process()
train_dataloader, test_dataloader, graph_data, train_grouped_data, test_grouped_data = get_values_by_keys(
    data_dict,
    keys=[
        'train_dataloader',
        'test_dataloader',
        'graph_data',
        'train_grouped_data',
        'test_grouped_data',
    ],
)
evaluation_manager = EvaluationManager(config=config, logger=logger)
train_manager: GraphNeuralNetworkTrainer = TrainManager(config=config, evaluation_manager=evaluation_manager).trainer
save_manager = SaveManager(config=config, logger=logger)

device = set_device(config['device'])
graph_data = graph_data.to(device)
model = NGCF(config=config, g=graph_data)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
model = model.to(device)
logger.send_message(config, message_type=0, message_content_type=0)
for i in range(config['epoch']):
    if config.get("early_stop", False):
        break
    train_metric = train_manager.train_model(
        model,
        train_dataloader,
        optimizer=optimizer,
        device=device,
        epoch=i,
    )
    #模型验证
    test_metric = None
    if i % 5 == 0 or i == config['epoch'] - 1:
        test_metric = train_manager.test_model(
            model,
            train_grouped_data,
            test_grouped_data,
            config['embedding_dim'] * (len(config['hidden_units']) + 1),
            i,
        )
    save_manager.save_all(
        epoch=i,
        train_metric=train_metric,
        valid_metric=None,
        test_metric=test_metric,
        other_metric=None,
        model=model,
    )
