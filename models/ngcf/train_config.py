'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-04-11
Description  : 
'''
#参数配置
train_config = {
    "model_name": "ngcf",
    "data": "gowalla",
    "debug_mode": False,
    "epoch": 10,
    "batch_size": 32,
    "lr": 0.0001,
    "device": -1,
    "embedding_dim": 64,
    "hidden_units": [64, 64, 64],
    "message_dropout": 0.1,
    "node_dropout": 0.1,
    "lmbd": 1e-5,
}
