'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-25
Description  : 
'''
import math
import numpy as np
from sklearn.metrics import roc_auc_score


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


def gauc(df, user_col, label_col, pre_col):
    preds = df.groupby(user_col)[pre_col].apply(list).to_dict()
    labels = df.groupby(user_col)[label_col].apply(list).to_dict()
    count_user = 0
    count_auc = 0
    for u in preds.keys():
        if np.sum(labels[u]) == 0 or np.sum(labels[u]) == len(labels[u]):
            # labels 里面全是1或者全是0，无法计算auc
            continue
        count_auc += len(labels[u]) * roc_auc_score(labels[u], preds[u])
        count_user += len(labels[u])
    return count_auc / count_user
