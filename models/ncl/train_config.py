'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-09-03
Description  : 
'''
#参数配置
train_config = {
    "model_name": "ncl",
    "data": "yelp2018",
    "trainer": "gnn",
    "debug_mode": False,
    "is_auto_gpu": True,
    # "is_saved": False,
    "eval_step": 1,
    "default": {
        "need_free_mem": 6000,
        "epoch": 1000,
        "batch_size": 2048,
        "lr": 0.001,
        "device": 0,
        "embedding_dim": 64,
        "hidden_units": [64, 64, 64],
        "hyper_layers": 1,
        "num_clusters": 1000,
        "lmbd_reg": 1e-4,
        "lmbd_ssl": 1e-7,
        "lmbd_proto": 8e-8,
        "cl_tau": 0.1,
        "alpha_ssl": 1,
        "alpha_proto": 1,
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
            "cl_tau": 0.2,
            "alpha_ssl": 1,
            "alpha_proto": 1,
            "lmbd_reg": 1e-4,
            "lmbd_ssl": 1e-6,
            "lmbd_proto": 1e-7,
            "num_clusters": 2000,
        },
        # "yelp2018": {
        #     "cl_tau": 0.05,
        #     "alpha_ssl": 1.5,
        #     "lmbd_reg": 1e-6,
        #     "lmbd_ssl": 1e-6,
        #     "lmbd_proto": 1e-7,
        #     "num_clusters": 500,
        # },
        "amazon_book": {
            "cl_tau": 0.05,
            "alpha_ssl": 0.8,
            "lmbd_reg": 1e-6,
            "lmbd_ssl": 1e-6,
            "lmbd_proto": 1e-7,
            "num_clusters": 2000,
        },
        "gowalla": {
            "cl_tau": 0.05,
            "alpha_ssl": 0.5,
            "lmbd_reg": 1e-4,
            "lmbd_ssl": 1e-6,
            "lmbd_proto": 5e-8,
            "num_clusters": 10
        },
        "ifashion": {},
    }
}
