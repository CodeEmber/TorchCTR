'''
Author       : wyx-hhhh
Date         : 2024-03-08
LastEditTime : 2024-03-08
Description  : 
'''
import os
import pandas as pd
import torch

from models.ncf.config import config
from models.ncf.model import NCF
from utils.evaluation import hitrate_dataframe, hitrate_dataloader, ndcg_dataframe, ndcg_dataloader
from utils.file_utils import get_file_path
from utils.preprocessing import ProcessData
from utils.save_utils import save_all
from utils.torch_utils import set_device
from trainers.movielens_train import get_test_predict, train_model, test_model
from utils.logger import MyLogger

logger = MyLogger()

Data = ProcessData(config=config)
config = Data.config
train_dataloader, valid_dataloader, test_dataloader, enc_dict = Data.data_process()

data_list = []

for batch in test_dataloader:
    # 假设每个 batch 包含 'user_id' 和 'label' 字段
    user_ids = batch['user_id'].numpy()
    labels = batch['label'].numpy()

    for user_id, label in zip(user_ids, labels):
        data_list.append({'user_id': user_id, 'label': label})

# 将列表转换为 DataFrame
test_df = pd.DataFrame(data_list)

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
hitrate = hitrate_dataframe(test_df)
ndcg = ndcg_dataframe(test_df)
logger.info(f"Hitrate: {hitrate}")
logger.info(f"NDCG: {ndcg}")
