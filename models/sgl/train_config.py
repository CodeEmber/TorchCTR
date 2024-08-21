'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-08-16
Description  : 
'''
#参数配置
train_config = {
    "model_name": "sgl",
    "data": "yelp2018",
    "trainer": "gnn",
    "debug_mode": False,
    "need_free_mem": 15000,
    "epoch": 1000,
    "batch_size": 2048,
    "lr": 0.001,
    "device": -1,
    "add_self_loop": True,
    "embedding_dim": 64,
    "hidden_units": [64, 64, 64],
    "lmbd_reg": 1e-4,
    "lmbd_ssl": 0.02,
    "aug_type": "ED",
    "dropout_ratio": 0.1,
    "ssl_tau": 0.2,
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
