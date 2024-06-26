'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-06-26
Description  : 
'''
#参数配置
train_config = {
    "model_name": "lightgcn",
    "data": "gowalla",
    "trainer": "gnn",
    "debug_mode": False,
    "epoch": 10,
    "batch_size": 32,
    "lr": 0.0001,
    "device": -1,
    "add_self_loop": False,
    "embedding_dim": 64,
    "hidden_units": [64, 64, 64],
    "lmbd": 1e-5,
    "metric_func": {
        "train": [
            {
                "eval_func": "loss",
            },
        ],
        "eval": [
            {
                "eval_func": "ndcg",
                "k": [10, 20]
            },
            {
                "eval_func": "recall",
                "k": [10, 20]
            },
        ],
    },
}
