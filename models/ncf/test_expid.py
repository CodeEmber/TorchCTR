'''
Author       : wyx-hhhh
Date         : 2024-03-08
LastEditTime : 2024-03-22
Description  : 
'''
import os
import pandas as pd
import torch

from models.ncf.config import config
from models.ncf.model import NCF
from utils.evaluation import hitrate, ndcg
from utils.file_utils import get_file_path
from data.data_manager import DataManager
from utils.save_utils import save_all
from utils.torch_utils import set_device
from trainers.movielens_train import get_test_predict
from utils.logger import logger
from utils.utilities import get_values_by_keys

DataManager = DataManager(config=config)
config = DataManager.config
data_dict = DataManager.data_process()
test_df, test_dataloader, enc_dict = get_values_by_keys(data_dict, ['test_df', 'test_dataloader', 'enc_dict'])

model = NCF(enc_dict=enc_dict)
device = set_device(config['device'])
model_file_path = get_file_path(['results', config["model_name"], 'save_model', f'{config["model_name"]}_{config["data"]}.pth'])
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"模型文件{model_file_path}不存在")
model.load_state_dict(torch.load(model_file_path))
model = model.to(device)

test_predictions = get_test_predict(model, test_dataloader, device)
test_df['prediction'] = test_predictions
test_df['ranking'] = test_df.groupby('user_id')['prediction'].rank(ascending=False, method='first')

logger.info("开始测试模型")
hitrate = hitrate(test_df)
ndcg = ndcg(test_df)
logger.info(f"Hitrate: {hitrate}")
logger.info(f"NDCG: {ndcg}")
