'''
Author       : wyx-hhhh
Date         : 2024-03-08
LastEditTime : 2024-06-12
Description  : 
'''
import os
import pandas as pd
import torch

from models.din.train_config import train_config
from models.din.model import DIN
from managers import ConfigManager, DataManager, LoggerManager, TrainManager, SaveManager, EvaluationManager
from utils.evaluation import gauc
from utils.file_utils import get_new_file_path
from utils.torch_utils import set_device
from managers.logger_manager import logger
from utils.utilities import get_values_by_keys

config_manager = ConfigManager(train_config=train_config)
config = config_manager.get_config()
logger = LoggerManager(config=config)
data_manager = DataManager(config=config)
data_dict = data_manager.data_process()
test_df, test_dataloader, enc_dict = get_values_by_keys(data_dict, ['test_df', 'test_dataloader', 'enc_dict'])
evaluation_manager = EvaluationManager(config=config, logger=logger)
train_manager = TrainManager(config=config).trainer

model = DIN(enc_dict=enc_dict)
device = set_device(config['device'])
model_file_path = get_new_file_path(['results', config["model_name"], 'save_model'])
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"模型文件{model_file_path}不存在")
model.load_state_dict(torch.load(model_file_path))
model = model.to(device)

test_predictions = train_manager.get_test_predict(model, test_dataloader, device)
test_df['prediction'] = test_predictions

gauc_result = evaluation_manager.get_eval_res(test_df=test_df, mode='eval')
logger.info(f"测试集GAUC: {gauc_result}")
