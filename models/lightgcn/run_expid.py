'''
Author       : wyx-hhhh
Date         : 2024-06-26
LastEditTime : 2024-06-28
Description  : 
'''
import torch

from models.lightgcn.model import LightGCN
from models.lightgcn.train_config import train_config
from managers import ConfigManager, DataManager, LoggerManager, TrainManager, SaveManager, EvaluationManager
from utils.torch_utils import set_device
from utils.utilities import get_values_by_keys
from trainers.gnn_trainer import GraphNeuralNetworkTrainer

config_manager = ConfigManager(train_config=train_config)
config = config_manager.get_config()
logger = LoggerManager(config=config)
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
model = LightGCN(config=config, g=graph_data)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
model = model.to(device)
for i in range(config['epoch']):
    train_metric = train_manager.train_model(model, train_dataloader, optimizer=optimizer, device=device)
    save_manager.save_all(
        epoch=i,
        train_metric=train_metric,
        model=model,
        is_clear=True,
    )
    logger.info(f"Epoch: {i + 1}")
    logger.info(f"Train Metric: {train_metric}")
    #模型验证
    if i % 10 == 0 or i == config['epoch'] - 1:
        test_metric = train_manager.test_model(
            model,
            train_grouped_data,
            test_grouped_data,
            config['embedding_dim'],
        )
        logger.info(f"Epoch {i} Test Metric: {test_metric}")
        logger.send_message(
            message={
                'Epoch': i,
                'Test Metric': test_metric
            },
            message_type=2,
            message_content_type=0,
        )
