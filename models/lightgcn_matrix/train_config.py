'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-08-20
Description  : 
'''
#参数配置
train_config = {
    "model_name": "lightgcn_matrix",
    # "data": "yelp2018",
    # "data": "amazon_book",
    "data": "gowalla_matrix",
    "trainer": "gnn",
    "debug_mode": False,
    "need_free_mem": 2000,
    "epoch": 1000,
    "batch_size": 2048,
    "lr": 0.001,
    "device": -1,
    "embedding_dim": 64,
    "hidden_units": [64, 64, 64],
    "lmbd": 0.0001,
    "is_split": False,
    "dropout": False,
    "keep_prob": 0.6,
    "add_self_loop": False,
    "metric_func": {
        "train": [
            {
                "eval_func": "loss",
            },
        ],
        "eval": [
            {
                "eval_func": "ndcg",
                "k": [20]
            },
            {
                "eval_func": "recall",
                "k": [20]
            },
            {
                "eval_func": "precision",
                "k": [20]
            },
        ],
    },
}
