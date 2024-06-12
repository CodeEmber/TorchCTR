'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-06-12
Description  : 
'''
#参数配置
train_config = {
    "model_name": "deepfm",
    "data": "criteo",
    "trainer": "dl",
    "debug_mode": True,
    "epoch": 5,
    "batch_size": 1024,
    "lr": 0.0001,
    "device": -1,
    "metric_func": {
        "train": [
            {
                "eval_func": "auc"
            },
            {
                "eval_func": "log_loss"
            },
        ],
        "valid": [
            {
                "eval_func": "auc"
            },
            {
                "eval_func": "log_loss"
            },
        ],
        "eval": [
            {
                "eval_func": "auc"
            },
            {
                "eval_func": "log_loss"
            },
        ],
    },
    "col_name": {
        "user_col": "user_id",
        "ranking_col": "ranking",
        "label_col": "label",
        "pre_col": "prediction",
    },
}
