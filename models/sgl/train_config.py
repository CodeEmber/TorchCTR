'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-07-09
Description  : 
'''
#参数配置
train_config = {
    "model_name": "sgl",
    "data": "gowalla",
    "trainer": "gnn",
    "debug_mode": False,
    "need_free_mem": 15000,
    "epoch": 800,
    "batch_size": 256,
    "lr": 0.0001,
    "device": -1,
    "add_self_loop": True,
    "embedding_dim": 64,
    "hidden_units": [64, 64, 64],
    "lmbd": 1e-5,
    "aug_type": "ED",
    "node_dropout_num": 0.2,
    "edge_dropout_num": 0.2,
    "ssl_tau": 0.1,
    "ssl_weight": 0.01,
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
