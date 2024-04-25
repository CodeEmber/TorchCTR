'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-04-12
Description  : 
'''
import torch

from models.ngcf.model import NGCF
from models.ngcf.train_config import train_config
from managers import ConfigManager, DataManager, LoggerManager, TrainManager, SaveManager
from utils.torch_utils import set_device
from utils.utilities import get_values_by_keys

ConfigManager = ConfigManager(train_config=train_config)
config = ConfigManager.get_config()
logger = LoggerManager(config=config)
DataManager = DataManager(config=config)
data_dict = DataManager.data_process()
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
TrainManager = TrainManager(config=config).trainer
SaveManager = SaveManager(config=config, logger=logger)

device = set_device(config['device'])
graph_data = graph_data.to(device)
model = NGCF(config=config, g=graph_data)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
model = model.to(device)
for i in range(config['epoch']):
    train_metric = TrainManager.train_model(model, train_dataloader, optimizer=optimizer, device=device)
    SaveManager.save_all(
        epoch=i,
        train_metric=train_metric,
        model=model,
        is_clear=True,
    )
    logger.info(f"Epoch: {i + 1}")
    logger.info(f"Train Metric: {train_metric}")
    #模型验证
    if i % 10 == 0 or i == config['epoch'] - 1:
        test_metric = TrainManager.test_model(
            model,
            train_grouped_data,
            test_grouped_data,
            device,
            config['embedding_dim'] * (len(config['hidden_size'])),
            20,
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
