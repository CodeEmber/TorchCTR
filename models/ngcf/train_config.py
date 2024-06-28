'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-06-28
Description  : 
'''
#参数配置
train_config = {
    "model_name": "ngcf",
    "data": "gowalla",
    "trainer": "gnn",
    "debug_mode": False,
    "epoch": 500,
    "batch_size": 32,
    "lr": 0.0001,
    "device": -1,
    "need_free_mem": 17000,
    "embedding_dim": 32,
    "hidden_units": [32, 32, 32],
    "message_dropout": 0.1,
    "node_dropout": 0.1,
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
