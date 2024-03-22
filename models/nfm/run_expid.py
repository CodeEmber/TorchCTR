'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-22
Description  : 
'''
import torch

from models.nfm.config import config
from models.nfm.model import NFM
from data.data_manager import DataManager
from utils.save_utils import save_all
from utils.torch_utils import set_device
from trainers.criteo_train import test_model, train_model, valid_model
from utils.logger import logger
from utils.utilities import get_values_by_keys

DataManager = DataManager(config=config)
config = DataManager.config
data_dict = DataManager.data_process()
train_dataloader, valid_dataloader, test_dataloader, enc_dict = get_values_by_keys(data_dict, ['train_dataloader', 'valid_dataloader', 'test_dataloader', 'enc_dict'])

model = NFM(enc_dict=enc_dict)
device = set_device(config['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
model = model.to(device)
for i in range(config['epoch']):
    train_metric = train_model(model, train_dataloader, optimizer=optimizer, device=device)
    valid_metric = valid_model(model, valid_dataloader, device)
    save_all(
        model_name=config['model_name'],
        data_name=config['data'],
        epoch=i,
        train_metric=train_metric,
        model=model,
        valid_metric=valid_metric,
        is_clear=True,
    )
    logger.info(f"Epoch: {i + 1}")
    logger.info(f"Train Metric: {train_metric}")
    logger.info(f"Valid Metric: {valid_metric}")

test_metric = test_model(model, test_dataloader, device)
logger.info(f"Test Metric: {test_metric}")
save_all(
    model_name=config['model_name'],
    data_name=config["data"],
    test_metric=test_metric,
    is_save_model=False,
    is_save_tensorboard=False,
)
