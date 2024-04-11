'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-04-11
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
train_dataloader, valid_dataloader, test_dataloader, graph_data = get_values_by_keys(data_dict, ['train_dataloader', 'valid_dataloader', 'test_dataloader', 'graph_data'])
TrainManager = TrainManager(config=config).trainer
SaveManager = SaveManager(config=config, logger=logger)

device = set_device(config['device'])
graph_data = graph_data.to(device)
model = NGCF(config=config, g=graph_data)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
model = model.to(device)
for i in range(config['epoch']):
    epoch_loss = TrainManager.train_model(model, train_dataloader, optimizer=optimizer, device=device)
    print(f"Epoch: {i + 1} Loss: {epoch_loss}")

# for i in range(config['epoch']):
#     train_metric = TrainManager.train_model(model, train_dataloader, optimizer=optimizer, device=device)
#     valid_metric = TrainManager.valid_model(model, valid_dataloader, device=device)
#     SaveManager.save_all(
#         epoch=i,
#         train_metric=train_metric,
#         valid_metric=valid_metric,
#         model=model,
#         is_clear=True,
#     )
#     logger.info(f"Epoch: {i + 1}")
#     logger.info(f"Train Metric: {train_metric}")

# test_metric = TrainManager.test_model(model, test_dataloader, device)
# logger.info(f"Test Metric: {test_metric}")
# SaveManager.save_all(
#     test_metric=test_metric,
#     is_save_model=False,
#     is_save_tensorboard=False,
# )
