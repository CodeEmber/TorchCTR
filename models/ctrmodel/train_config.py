'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-09-10
Description  : 
'''
#参数配置
train_config = {
    "model_name": "ctrmodel",
    "data": "yelp2018",
    "trainer": "gnn",
    "debug_mode": False,
    "is_auto_gpu": True,
    "is_saved": False,
    "eval_step": 1,
    "default": {
        "need_free_mem": 3000,
        "epoch": 1000,
        "batch_size": 2048,
        "lr": 0.001,
        "device": 3,
        "embedding_dim": 64,
        "hidden_units": [64, 64],
        "lmbd_reg": 1e-4,
        "lmbd_ssl": 0.2,
        "eps": 0.2,
        "cl_tau": 0.15,
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
        "yelp2018": {},
        "amazon_book": {
            "lmbd_ssl": 1,
        },
        "amazon_kindle": {
            "lmbd_ssl": 0.02,
            "eps": 0.1,
            "hidden_units": [64, 64, 64]
        },
        "gowalla": {
            "lmbd_ssl": 2,
        },
        "alibaba_ifashion": {
            "lmbd_ssl": 0.05,
            "eps": 0.05,
            "hidden_units": [64, 64, 64, 64],
        },
    }
}
