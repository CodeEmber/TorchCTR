'''
Author       : wyx-hhhh
Date         : 2024-03-25
LastEditTime : 2024-09-04
Description  : 
'''
DATA_CONFIG = {
    "gowalla": {
        "train_path": ['dataset', 'gowalla', 'train.txt'],
        "test_path": ['dataset', 'gowalla', 'test.txt'],
        "s_pre_adj_mat": ['dataset', 'gowalla', 's_pre_adj_mat.npz'],
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "amazon_book": {
        "train_path": ['dataset', 'amazon_book', 'train.txt'],
        "test_path": ['dataset', 'amazon_book', 'test.txt'],
        "s_pre_adj_mat": ['dataset', 'amazon_book', 's_pre_adj_mat.npz'],
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "yelp2018": {
        "train_path": ['dataset', 'yelp2018', 'train.txt'],
        "test_path": ['dataset', 'yelp2018', 'test.txt'],
        "s_pre_adj_mat": ['dataset', 'yelp2018', 's_pre_adj_mat.npz'],
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "alibaba_ifashion": {
        "train_path": ['dataset', 'alibaba_ifashion', 'train.txt'],
        "test_path": ['dataset', 'alibaba_ifashion', 'test.txt'],
        "s_pre_adj_mat": ['dataset', 'alibaba_ifashion', 's_pre_adj_mat.npz'],
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "amazon_kindle": {
        "train_path": ['dataset', 'amazon_kindle', 'train.txt'],
        "test_path": ['dataset', 'amazon_kindle', 'test.txt'],
        "s_pre_adj_mat": ['dataset', 'amazon_kindle', 's_pre_adj_mat.npz'],
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
    "douban_book": {
        "train_path": ['dataset', 'douban_book', 'train.txt'],
        "test_path": ['dataset', 'douban_book', 'test.txt'],
        "s_pre_adj_mat": ['dataset', 'douban_book', 's_pre_adj_mat.npz'],
        "col_name": {
            "user_col": "user_id",
            "label_col": "label",
            "pre_col": "prediction",
        },
    },
}
