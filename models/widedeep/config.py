'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2023-10-30
Description  : 
'''
#参数配置
config = {
    "model_name": "widedeep",
    "data_path": ['data', 'criteo.csv'],
    "sparse_cols": [f'C{x}' for x in range(1, 27)],
    "dense_cols": [f'I{x}' for x in range(1, 14)],
    "train_ratio": 0.7,
    "valid_ratio": 0.2,
    "debug_mode": True,
    "epoch": 5,
    "batch_size": 1024,
    "lr": 0.0001,
    "device": -1,
}
