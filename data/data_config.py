'''
Author       : wyx-hhhh
Date         : 2024-02-27
LastEditTime : 2024-02-27
Description  : 
'''
DATA_CONFIG = {
    "criteo": {
        "data_path": ['data', 'criteo', 'criteo.csv'],
        "sparse_cols": [f'C{x}' for x in range(1, 27)],
        "dense_cols": [f'I{x}' for x in range(1, 14)],
    },
    "movielens": {
        "data_path": ['data', 'movielens', 'movielens.csv'],
        "sparse_cols": ['userId', 'movieId'],
        "dense_cols": ['rating'],
    },
}
