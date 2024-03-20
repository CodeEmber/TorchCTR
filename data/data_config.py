'''
Author       : wyx-hhhh
Date         : 2024-02-27
LastEditTime : 2024-03-19
Description  : 
'''
DATA_CONFIG = {
    "criteo": {
        "data_path": ['data', 'criteo', 'criteo.csv'],
        "sparse_cols": [f'C{x}' for x in range(1, 27)],
        "dense_cols": [f'I{x}' for x in range(1, 14)],
    },
    "movielens": {
        "behaviour_path": ['data', 'movielens', 'ml-1m.inter'],
        "user_path": ['data', 'movielens', 'ml-1m.user'],
        "item_path": ['data', 'movielens', 'ml-1m.item'],
        "sparse_cols": ['user_id', 'item_id'],
        "dense_cols": [],
    },
    "amazon": {
        "behaviour_path": ['data', 'amazon', 'Amazon_Electronics.inter'],
        "item_path": ['data', 'amazon', 'Amazon_Electronics.item'],
        "sparse_cols": ['user_id', 'item_target_id', 'item_target_category'],
        "history_cols": ['item_history_seq_id', 'item_history_seq_category'],
    },
}
