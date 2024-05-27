'''
Author       : wyx-hhhh
Date         : 2024-05-27
LastEditTime : 2024-05-27
Description  : 
'''
import torch

from models.afm.model import AFM
from models.afm.train_config import train_config
from managers import ConfigManager, DataManager, LoggerManager, TrainManager, SaveManager, EvaluationManager
from utils.torch_utils import set_device
from utils.utilities import get_values_by_keys

config_manager = ConfigManager(train_config=train_config)
config = config_manager.get_config()
logger = LoggerManager(config=config)
data_manager = DataManager(config=config)
data_dict = data_manager.data_process()
train_dataloader, valid_dataloader, test_dataloader, enc_dict = get_values_by_keys(data_dict, ['train_dataloader', 'valid_dataloader', 'test_dataloader', 'enc_dict'])
evaluation_manager = EvaluationManager(config=config, logger=logger)
train_manager = TrainManager(config=config, evaluation_manager=evaluation_manager).trainer
save_manager = SaveManager(config=config, logger=logger)

model = AFM(enc_dict=enc_dict)

device = set_device(config['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
model = model.to(device)
for i in range(config['epoch']):
    train_metric = train_manager.train_model(model, train_dataloader, optimizer=optimizer, device=device)
    valid_metric = train_manager.valid_model(model, valid_dataloader, device=device)
    save_manager.save_all(
        epoch=i,
        train_metric=train_metric,
        valid_metric=valid_metric,
        model=model,
        is_clear=True,
    )
    logger.info(f"Epoch: {i + 1}")
    logger.info(f"Train Metric: {train_metric}")

test_metric = train_manager.test_model(model, test_dataloader, device)
logger.info(f"Test Metric: {test_metric}")
save_manager.save_all(
    test_metric=test_metric,
    is_save_model=False,
    is_save_tensorboard=False,
)
