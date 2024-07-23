'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-07-13
Description  : 
'''
#参数配置
train_config = {
    "model_name": "lightgcn_dgl",
    "data": "gowalla_original",
    "trainer": "gnn",
    "debug_mode": False,
    "need_free_mem": 10000,
    "epoch": 1000,
    "batch_size": 2048,
    "lr": 0.0001,
    "device": -1,
    "add_self_loop": True,
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
                "k": [20, 40]
            },
            {
                "eval_func": "recall",
                "k": [20, 40]
            },
        ],
    },
}
