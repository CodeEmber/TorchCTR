'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-09-04
Description  : 
'''
#参数配置
train_config = {
    "model_name": "ngcf",
    "data": "gowalla",
    # "is_saved": False,
    "trainer": "gnn",
    "debug_mode": False,
    "eval_step": 1,
    "default": {
        "epoch": 400,
        "batch_size": 4096,
        "lr": 0.0001,
        "device": 3,
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
                    "k": [10, 20, 50]
                },
                {
                    "eval_func": "recall",
                    "k": [10, 20, 50]
                },
                {
                    "eval_func": "precision",
                    "k": [10, 20, 50]
                },
                {
                    "eval_func": "hitrate",
                    "k": [10, 20, 50]
                },
            ],
        },
    },
    "data_specific_config": {
        "yelp2018": {
            "epoch": 1000,
        },
        "amazon_book": {
            "lr": 0.0005,
        },
        "gowalla": {
            "epoch": 1000,
        }
    }
}
