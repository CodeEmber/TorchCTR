'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-09-10
Description  : 
'''
#参数配置
train_config = {
    "model_name": "lightgcn",
    "trainer": "gnn",
    "data": "yelp2018",
    "debug_mode": False,
    # "is_saved": False,
    # "is_auto_gpu": True,
    "eval_step": 1,
    "default": {
        "need_free_mem": 2000,
        "epoch": 1000,
        "batch_size": 4096,
        "lr": 0.001,
        "device": 3,
        "embedding_dim": 64,
        "hidden_units": [64, 64, 64],
        "reg_lmbd": 0.0001,
        "is_split": False,
        "dropout": False,
        "keep_prob": 0.6,
        "add_self_loop": False,
        "neg_ratio": 1,
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
        "amazon_book": {},
        "gowalla": {}
    }
}
