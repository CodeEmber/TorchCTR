'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-22
Description  : 
'''
import math
import numpy as np
import pandas as pd


def hitrate(test_df, k=20):
    user_num = test_df['user_id'].nunique()
    test_gd_df = test_df[test_df['ranking'] <= k].reset_index(drop=True)
    return test_gd_df['label'].sum() / user_num


def ndcg(test_df, k=20):
    user_num = test_df['user_id'].nunique()
    test_gd_df = test_df[test_df['ranking'] <= k].reset_index(drop=True)
    test_gd_df['dcg'] = test_gd_df['label'] / np.log2(test_gd_df['ranking'] + 1)
    test_gd_df['idcg'] = 1 / np.log2(test_gd_df['ranking'] + 1)
    test_gd_df['ndcg'] = test_gd_df['dcg'] / test_gd_df['idcg']
    return test_gd_df['ndcg'].sum() / user_num
