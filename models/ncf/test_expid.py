'''
Author       : wyx-hhhh
Date         : 2024-03-08
LastEditTime : 2024-03-25
Description  : 
'''
import os
import pandas as pd
import torch

from models.ncf.train_config import train_config
from models.ncf.model import NCF
from managers import ConfigManager, DataManager, LoggerManager, TrainManager, SaveManager
from utils.evaluation import hitrate, ndcg
from utils.file_utils import get_new_file_path
from utils.torch_utils import set_device
from managers.logger_manager import logger
from utils.utilities import get_values_by_keys

ConfigManager = ConfigManager(train_config=train_config)
config = ConfigManager.get_config()
logger = LoggerManager(config=config)
DataManager = DataManager(config=config)
data_dict = DataManager.data_process()
test_df, test_dataloader, enc_dict = get_values_by_keys(data_dict, ['test_df', 'test_dataloader', 'enc_dict'])
TrainManager = TrainManager(config=config).trainer

model = NCF(enc_dict=enc_dict)
device = set_device(config['device'])
model_file_path = get_new_file_path(['results', config["model_name"], 'save_model'])
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"模型文件{model_file_path}不存在")
model.load_state_dict(torch.load(model_file_path))
model = model.to(device)

test_predictions = TrainManager.get_test_predict(model, test_dataloader, device)
test_df['prediction'] = test_predictions
test_df['ranking'] = test_df.groupby('user_id')['prediction'].rank(ascending=False, method='first')

logger.info("开始测试模型")
hitrate = hitrate(test_df)
ndcg = ndcg(test_df)
logger.info(f"Hitrate: {hitrate}")
logger.info(f"NDCG: {ndcg}")
