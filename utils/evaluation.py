'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-08
Description  : 
'''
import math
import numpy as np
import pandas as pd


def hitrate_dataframe(test_df, k=20):
    user_num = test_df['user_id'].nunique()
    test_gd_df = test_df[test_df['ranking'] <= k].reset_index(drop=True)
    return test_gd_df['label'].sum() / user_num


def ndcg_dataframe(test_df, k=20):
    user_num = test_df['user_id'].nunique()
    test_gd_df = test_df[test_df['ranking'] <= k].reset_index(drop=True)
    test_gd_df['dcg'] = test_gd_df['label'] / np.log2(test_gd_df['ranking'] + 1)
    test_gd_df['idcg'] = 1 / np.log2(test_gd_df['ranking'] + 1)
    test_gd_df['ndcg'] = test_gd_df['dcg'] / test_gd_df['idcg']
    return test_gd_df['ndcg'].sum() / user_num


def hitrate_dataloader(test_dataloader, test_predictions, k=20):
    total_users = 0
    total_hits = 0

    for batch in test_dataloader:
        true_labels = batch[0]['label'].numpy()
        for true_label, prediction in zip(true_labels, test_predictions):
            sorted_labels = true_label[np.argsort(-prediction)]
            hits_batch = np.sum(sorted_labels[:k])
            total_hits += hits_batch
            total_users += len(true_label)

    avg_hitrate = total_hits / total_users
    return avg_hitrate


def ndcg_dataloader(test_dataloader, test_predictions, k=20):
    total_users = 0
    total_ndcg = 0

    for true_labels, predictions in zip(test_dataloader, test_predictions):
        ndcg_batch = 0
        for true_label, prediction in zip(true_labels, predictions):
            sorted_labels = true_label[np.argsort(-prediction)]  # 按预测结果对真实标签排序
            dcg = np.sum(sorted_labels[:k] / np.log2(np.arange(2, min(k + 2, len(sorted_labels)) + 1)))  # 计算DCG
            ideal_sorted_labels = np.sort(true_label)[::-1]  # 理想情况下的排序
            idcg = np.sum(ideal_sorted_labels[:k] / np.log2(np.arange(2, min(k + 2, len(ideal_sorted_labels)) + 1)))  # 计算IDCG
            ndcg_user = dcg / idcg if idcg > 0 else 0
            total_ndcg += ndcg_user
            total_users += 1

    avg_ndcg = total_ndcg / total_users
    return avg_ndcg
