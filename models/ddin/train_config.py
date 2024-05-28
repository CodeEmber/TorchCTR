'''
Author       : wyx-hhhh
Date         : 2024-04-24
LastEditTime : 2024-05-28
Description  : 
'''
#参数配置
train_config = {
    "model_name": "ddin",
    "data": "pixelrec",
    "trainer": "dl",
    "debug_mode": True,
    "epoch": 10,
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
        "eval": [
            {
                "eval_func": "ndcg",
                "k": [10]
            },
            {
                "eval_func": "gauc"
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
