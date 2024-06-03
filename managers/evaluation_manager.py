'''
Author       : wyx-hhhh
Date         : 2024-05-22
LastEditTime : 2024-06-03
Description  : 
'''

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from managers.logger_manager import LoggerManager


class EvaluationManager():

    def __init__(
        self,
        config: dict,
        logger: LoggerManager = None,
    ):
        self.config = config
        # 传入 > 配置 > 默认
        if config.get("metric_func") is None:
            raise ValueError("metric_func is None, config must have metric_func")
        self.metric_func = config.get("metric_func")
        self.col_name = config.get("col_name")
        self.logger = logger

    def logloss(self, y_true, y_pred):
        return log_loss(np.concatenate(y_true), np.concatenate(y_pred), eps=1e-7)

    def auc(self, y_true, y_pred):
        return roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))

    def hitrate(self, test_df, user_col, ranking_col, label_col, k=20):
        user_num = test_df[user_col].nunique()
        test_gd_df = test_df[test_df[ranking_col] <= k].reset_index(drop=True)
        return test_gd_df[label_col].sum() / user_num

    def ndcg(self, test_df, user_col, ranking_col, label_col, k=20):
        user_num = test_df[user_col].nunique()
        test_gd_df = test_df[test_df[ranking_col] <= k].reset_index(drop=True)
        test_gd_df['dcg'] = test_gd_df[label_col] / np.log2(test_gd_df[ranking_col] + 1)
        test_gd_df['idcg'] = 1 / np.log2(test_gd_df[ranking_col] + 1)
        test_gd_df['ndcg'] = test_gd_df['dcg'] / test_gd_df['idcg']
        return test_gd_df['ndcg'].sum() / user_num

    def gauc(self, test_df, user_col, label_col, pre_col):
        preds = test_df.groupby(user_col)[pre_col].apply(list).to_dict()
        labels = test_df.groupby(user_col)[label_col].apply(list).to_dict()
        count_user = 0
        count_auc = 0
        for u in preds.keys():
            if np.sum(labels[u]) == 0 or np.sum(labels[u]) == len(labels[u]):
                continue
            count_auc += len(labels[u]) * roc_auc_score(labels[u], preds[u])
            count_user += len(labels[u])
        return count_auc / count_user

    def get_eval_res(
        self,
        y_true=None,
        y_pred=None,
        test_df=None,
        mode="train",
        metric_func=None,
        col_name=None,
    ):
        res_dict = dict()
        if metric_func is not None:
            self.metric_func = metric_func
        if col_name is not None:
            self.col_name = col_name
        try:
            if mode == "train":
                for metric in self.metric_func.get("train"):
                    if metric["eval_func"] == "auc":
                        if y_true or y_pred:
                            res_dict["auc"] = self.auc(y_true, y_pred)
                        else:
                            self.logger.info("y_true和y_pred为空，无法计算auc，可能是因为bach_size过大，数据过少，同时设置了drop_last=True")
                    elif metric["eval_func"] == "log_loss":
                        if y_true or y_pred:
                            res_dict["log_loss"] = self.logloss(y_true, y_pred)
                        else:
                            self.logger.info("y_true和y_pred为空，无法计算log_loss，可能是因为bach_size过大，数据过少，同时设置了drop_last=True")

                    else:
                        raise ValueError("eval_func error")
            elif mode == "eval":
                for metric in self.metric_func.get("eval"):
                    if metric["eval_func"] == "hitrate":
                        for k in metric["k"]:
                            res_dict[f"hitrate@{k}"] = self.hitrate(test_df, self.col_name.get("user_col"), self.col_name.get("ranking_col"), self.col_name.get("label_col"), k)
                    elif metric["eval_func"] == "ndcg":
                        for k in metric["k"]:
                            res_dict[f"ndcg@{k}"] = self.ndcg(test_df, self.col_name.get("user_col"), self.col_name.get("ranking_col"), self.col_name.get("label_col"), k)
                    elif metric["eval_func"] == "gauc":
                        res_dict["gauc"] = self.gauc(test_df, self.col_name.get("user_col"), self.col_name.get("label_col"), self.col_name.get("pre_col"))
                    else:
                        raise ValueError("eval_func error")
            else:
                raise ValueError("mode error")
            return res_dict
        except:
            raise ValueError("metric_func error")
