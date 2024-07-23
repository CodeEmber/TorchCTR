'''
Author       : wyx-hhhh
Date         : 2024-03-25
LastEditTime : 2024-07-18
Description  : 
'''
DATA_CONFIG = {
    "criteo": {
        "data_path": ['data', 'criteo', 'criteo.csv'],
        "sparse_cols": [f'C{x}' for x in range(1, 27)],
        "dense_cols": [f'I{x}' for x in range(1, 14)],
        "train_ratio": 0.7,
        "valid_ratio": 0.2,
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "movielens": {
        "behaviour_path": ['data', 'movielens', 'ml-1m.inter'],
        "user_path": ['data', 'movielens', 'ml-1m.user'],
        "item_path": ['data', 'movielens', 'ml-1m.item'],
        "sparse_cols": ['user_id', 'item_id'],
        "dense_cols": [],
        "neg_sample_ratio": 3,
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "amazon": {
        "behaviour_path": ['data', 'amazon', 'Amazon_Electronics.inter'],
        "item_path": ['data', 'amazon', 'Amazon_Electronics.item'],
        "sparse_cols": ['user_id', 'item_target_id', 'item_target_category'],
        "history_cols": ['item_history_seq_id', 'item_history_seq_category'],
        "neg_sample_ratio": 2,
        "train_ratio": 0.8,
        "valid_ratio": 0.1,
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "gowalla_dgl": {
        "origin_path": ['data', 'gowalla_original', 'loc-gowalla_totalCheckins.txt'],
        "data_path": ['data', 'gowalla_original', 'gowalla.csv'],
        "train_path": ['data', 'gowalla_original', 'train.csv'],
        "test_path": ['data', 'gowalla_original', 'test.csv'],
        "sparse_cols": ['user_id', 'item_id'],
        "train_ratio": 0.8,
        "user_num": 66447,
        "item_num": 1242326,
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "gowalla_matrix": {
        "train_path": ['data', 'gowalla_matrix', 'train.txt'],
        "test_path": ['data', 'gowalla_matrix', 'test.txt'],
        "user_list_path": ['data', 'gowalla_matrix', 'user_list.txt'],
        "item_list_path": ['data', 'gowalla_matrix', 'item_list.txt'],
        "s_pre_adj_mat": ['data', 'gowalla_matrix', 's_pre_adj_mat.npz'],
        "user_num": 66447,
        "item_num": 1242326,
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "pixelrec": {
        "behaviour_path": ['data', 'pixelrec', 'Pixel200K_inter.csv'],
        "item_path": ['data', 'pixelrec', 'Pixel200K_item.csv'],
        "sparse_cols": ['user_id', 'item_target_id', 'item_target_tag'],
        "history_cols": ['item_history_seq_id', 'item_history_seq_tag'],
        "neg_sample_ratio": 2,
        "train_ratio": 0.8,
        "valid_ratio": 0.1,
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
}
