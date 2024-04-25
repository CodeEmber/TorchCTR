'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-04-12
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


def evaluate_recall(preds, test_gd, top_n=50):
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for user in test_gd.keys():
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:top_n]:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    return {f'recall@{top_n}': round(recall, 4), f'ndcg@{top_n}': round(ndcg, 4), f'hitrate@{top_n}': round(hitrate, 4)}
