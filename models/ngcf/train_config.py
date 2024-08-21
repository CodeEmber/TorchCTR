'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-08-16
Description  : 
'''
#参数配置
train_config = {
    "model_name": "ngcf",
    # "data": "yelp2018",
    # "data": "amazon_book",
    "data": "gowalla_matrix",
    "trainer": "gnn",
    "debug_mode": False,
    "epoch": 400,
    "batch_size": 4096,
    "lr": 0.0001,
    "device": -1,
    "need_free_mem": 2000,
    "embedding_dim": 64,
    "hidden_units": [64, 64, 64],
    "message_dropout_ratio": 0.1,
    "node_dropout_ratio": 0.1,
    "decay": 1e-5,
    "metric_func": {
        "train": [
            {
                "eval_func": "loss",
            },
        ],
        "eval": [
            {
                "eval_func": "ndcg",
                "k": [20, 40]
            },
            {
                "eval_func": "recall",
                "k": [20, 40]
            },
        ],
    },
}
