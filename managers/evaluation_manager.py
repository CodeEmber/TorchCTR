'''
Author       : wyx-hhhh
Date         : 2024-05-22
LastEditTime : 2024-06-28
Description  : 
'''

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from managers.logger_manager import LoggerManager


class EvaluationManager():

    def __init__(
        self,
        config: dict = {},
        logger: LoggerManager = {},
    ):
        self.config = config
        self.metric_func = config.get("metric_func", dict())
        self.col_name = config.get("col_name", dict())
        self.logger = logger

    def logloss(self, y_true, y_pred):
        return log_loss(np.concatenate(y_true), np.concatenate(y_pred), eps=1e-7)

    def auc(self, y_true, y_pred):
        return roc_auc_score(np.concatenate(y_true), np.concatenate(y_pred))

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((np.concatenate(y_true) - np.concatenate(y_pred))**2))

    def precision(self, test_df: pd.DataFrame, col_name: dict, k: int = 20):
        """Precision@k

        Args:
            test_df (pd.DataFrame): 测试集
            col_name (dict): 列名，包含user_col, pre_col, label_col
            k (int, optional): 排序数量. Defaults to 20.

        Returns:
            precision (int): 平均准确率
        """
        precisions = []
        for _, user_group in test_df.groupby(col_name['user_col']):
            if 'ranking' not in test_df.columns:
                test_df['ranking'] = test_df.groupby(col_name['user_col'])[col_name['pre_col']].rank(ascending=False, method='first')
            user_relevant_items = user_group[user_group['ranking'] <= k]
            precision = user_relevant_items[col_name['label_col']].sum() / k
            precisions.append(precision)
        return sum(precisions) / len(precisions)

    def hitrate(self, test_df: pd.DataFrame, col_name: dict, k: int = 20):
        """基于命中用户的评估指标
        
        Args:
            test_df (pd.DataFrame): 测试集
            col_name (dict): 列名，包含user_col, pre_col, label_col
            k (int, optional): 排序数量. Defaults to 20.
        
        Returns:
            hitrate (int): 命中用户数量比例
        """
        if 'ranking' not in test_df.columns:
            test_df['ranking'] = test_df.groupby(col_name['user_col'])[col_name['pre_col']].rank(ascending=False, method='first')
        test_gd_df = test_df[test_df['ranking'] <= k]
        test_gd_df = test_gd_df[test_gd_df[col_name['label_col']] == 1].reset_index(drop=True)
        return test_gd_df[col_name['user_col']].nunique() / test_df[col_name['user_col']].nunique()

    def mrr(self, test_df: pd.DataFrame, col_name: dict, k: int = 20):
        """Mean Reciprocal Rank
        
        Args:
            test_df (pd.DataFrame): 测试集
            col_name (dict): 列名，包含user_col, pre_col, label_col
            k (int, optional): 排序数量. Defaults to 20.
        
        Returns:
            mrr (int): 平均倒数排名
        """
        if 'ranking' not in test_df.columns:
            test_df['ranking'] = test_df.groupby(col_name['user_col'])[col_name['pre_col']].rank(ascending=False, method='first')
        test_gd_df = test_df[(test_df['ranking'] <= k) & (test_df[col_name['label_col']] == 1)].reset_index(drop=True)
        test_gd_df = test_gd_df.sort_values(by=[col_name['user_col'], 'ranking'], ascending=[True, True])
        test_gd_df = test_gd_df.drop_duplicates(subset=[col_name['user_col']], keep='first').reset_index(drop=True)
        test_gd_df['mrr'] = 1 / test_gd_df['ranking']
        return test_gd_df['mrr'].sum() / test_df[col_name['user_col']].nunique()

    def ndcg(self, test_df: pd.DataFrame, col_name: dict, k: int = 20, version: int = 0):
        """Normalized Discounted Cumulative Gain
        
        Args:
            test_df (pd.DataFrame): 测试集
            col_name (dict): 列名，包含user_col, pre_col, label_col
            k (int, optional): 排序数量. Defaults to 20.
            version (int, optional): dcg版本. Defaults to 0:rel_i/log2(i+1);1:(2^rel_i-1)/log2(i+1)
            
        Returns:
            ndcg (int): 平均ndcg
        """
        if 'ranking' not in test_df.columns:
            test_df['ranking'] = test_df.groupby(col_name['user_col'])[col_name['pre_col']].rank(ascending=False, method='first')
        test_gd_df = test_df[(test_df['ranking'] <= k) & (test_df[col_name['label_col']] == 1)].reset_index(drop=True)
        ndcg_values = []  # 保存每个用户的dcg和idcg
        for user_id, user_group in test_gd_df.groupby(col_name['user_col']):
            if version == 0:
                dcg = sum(user_group[col_name['label_col']] / np.log2(user_group['ranking'] + 1))
            else:
                dcg = sum((2**user_group[col_name['label_col']] - 1) / np.log2(user_group['ranking'] + 1))
            labels_sorted = user_group[col_name['label_col']].sort_values(ascending=False)
            if version == 0:
                idcg = sum(labels_sorted / np.log2(range(2, len(labels_sorted) + 2)))
            else:
                idcg = sum((2**labels_sorted - 1) / np.log2(range(2, len(labels_sorted) + 2)))
            ndcg_values.append((user_id, dcg, idcg))
        ndcg_values = np.array(ndcg_values)
        if len(ndcg_values) == 0:
            return 0
        ndcg_values = ndcg_values[ndcg_values[:, 2] != 0]  # 防止出现0除
        return np.mean(ndcg_values[:, 1] / ndcg_values[:, 2])

    def recall(self, test_df: pd.DataFrame, col_name: dict, k: int = 20):
        """Recall@k
    
        Args:
            test_df (pd.DataFrame): 测试集
            col_name (dict): 列名，包含user_col, pre_col, label_col
            k (int, optional): 排序数量. Defaults to 20.
    
        Returns:
            recall (int): 平均召回率
        """
        recalls = []
        for _, user_group in test_df.groupby(col_name['user_col']):
            if 'ranking' not in test_df.columns:
                test_df['ranking'] = test_df.groupby(col_name['user_col'])[col_name['pre_col']].rank(ascending=False, method='first')
            user_relevant_items = user_group[user_group['ranking'] <= k]
            denominator = user_group[col_name['label_col']].sum()
            if denominator != 0:
                recall = user_relevant_items[col_name['label_col']].sum() / denominator
            else:
                recall = 0
            recalls.append(recall)
        return sum(recalls) / len(recalls)

    def Fscore(self, test_df: pd.DataFrame, col_name: dict, k: int = 20, beta: int = 1):
        """F-score@k

        Args:
            test_df (pd.DataFrame): 测试集
            col_name (dict): 列名，包含user_col, pre_col, label_col
            k (int, optional): 排序数量. Defaults to 20.
            beta (int, optional): beta值. Defaults to 1.

        Returns:
            fscore (int): 平均F-score
        """
        fscores = []
        for _, user_group in test_df.groupby(col_name['user_col']):
            if 'ranking' not in test_df.columns:
                test_df['ranking'] = test_df.groupby(col_name['user_col'])[col_name['pre_col']].rank(ascending=False, method='first')
            user_relevant_items = user_group[user_group['ranking'] <= k]
            precision = user_relevant_items[col_name['label_col']].sum() / k
            recall = user_relevant_items[col_name['label_col']].sum() / user_group[col_name['label_col']].sum()
            if precision + recall == 0:
                fscores.append(0)
            else:
                fscores.append((1 + beta**2) * precision * recall / (beta**2 * precision + recall))
        return sum(fscores) / len(fscores)

    def gauc(self, test_df: pd.DataFrame, col_name: dict):
        """Grouped AUC
        
        Args:
            test_df (pd.DataFrame): 测试集
            col_name (dict): 列名，包含user_col, label_col, pre_col
            
        Returns:
            gauc (int): 平均gauc
            
        Note:
            - 该函数适用于用户行为序列数据
        """
        preds = test_df.groupby(col_name['user_col'])[col_name['pre_col']].apply(list).to_dict()
        labels = test_df.groupby(col_name['user_col'])[col_name['label_col']].apply(list).to_dict()
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
        mode="train",
        metric_func=None,
        y_true=None,
        y_pred=None,
        loss=None,
        test_df=None,
        col_name=None,
    ):
        """输出评估结果

        Args:
            mode (str, optional): 模式. Defaults to "train".
            y_true (list, optional): 真实值. Defaults to None.
            y_pred (list, optional): 预测值. Defaults to None.
            test_df (pd.DataFrame, optional): 测试集. Defaults to None.
            metric_func (dict, optional): 评估函数. Defaults to None.
            col_name (dict, optional): 列名. Defaults to None.
            loss (float, optional): 损失. Defaults to None.
        
        Returns:
            dict: 评估结果
            
        Raises:
            ValueError: metric_func error
            ValueError: mode error
            ValueError: eval_func error
            ValueError: loss is None
            ValueError: y_true和y_pred为空，无法计算auc，可能是因为bach_size过大，数据过少，同时设置了drop_last=True
            
        Note:
            - metric_func包含auc或log_loss需要传入y_true和y_pred.  
            - metric_func包含loss需要传入loss.
            - metric_func包含hitrate或ndcg需要传入test_df和col_name. 
        """
        res_dict = dict()
        if metric_func is not None:
            self.metric_func = metric_func
        if col_name is not None:
            self.col_name = col_name
        try:
            for metric_set in ["train", "valid", "eval"]:
                if mode == metric_set:
                    for metric in self.metric_func.get(metric_set):
                        if metric["eval_func"] == "hitrate":
                            for k in metric.get("k", [20]):
                                res_dict[f"hitrate@{k}"] = self.hitrate(test_df=test_df, col_name=self.col_name, k=k)
                        elif metric["eval_func"] == "fscore":
                            for k in metric.get("k", [20]):
                                res_dict[f"fscore@{k}"] = self.Fscore(test_df=test_df, col_name=self.col_name, k=k)
                        elif metric["eval_func"] == "precision":
                            for k in metric.get("k", [20]):
                                res_dict[f"precision@{k}"] = self.precision(test_df=test_df, col_name=self.col_name, k=k)
                        elif metric["eval_func"] == "recall":
                            for k in metric.get("k", [20]):
                                res_dict[f"recall@{k}"] = self.recall(test_df=test_df, col_name=self.col_name, k=k)
                        elif metric["eval_func"] == "mrr":
                            for k in metric.get("k", [20]):
                                res_dict[f"mrr@{k}"] = self.mrr(test_df=test_df, col_name=self.col_name, k=k)
                        elif metric["eval_func"] == "ndcg":
                            for k in metric.get("k", [20]):
                                res_dict[f"ndcg@{k}"] = self.ndcg(
                                    test_df=test_df,
                                    col_name=self.col_name,
                                    k=k,
                                    version=metric.get("version", 0),
                                )
                        elif metric["eval_func"] == "gauc":
                            res_dict["gauc"] = self.gauc(test_df=test_df, col_name=self.col_name)
                        elif metric["eval_func"] == "auc":
                            if y_true or y_pred:
                                res_dict["auc"] = self.auc(y_true, y_pred)
                            else:
                                self.logger.info("y_true和y_pred为空，无法计算auc，可能是因为bach_size过大，数据过少，同时设置了drop_last=True")
                        elif metric["eval_func"] == "log_loss":
                            if y_true or y_pred:
                                res_dict["log_loss"] = self.logloss(y_true, y_pred)
                            else:
                                self.logger.info("y_true和y_pred为空，无法计算log_loss，可能是因为bach_size过大，数据过少，同时设置了drop_last=True")
                        elif metric["eval_func"] == "rmse":
                            if y_true or y_pred:
                                res_dict["rmse"] = self.rmse(y_true, y_pred)
                            else:
                                self.logger.info("y_true和y_pred为空，无法计算rmse，可能是因为bach_size过大，数据过少，同时设置了drop_last=True")
                        elif metric["eval_func"] == "loss":
                            if loss:
                                res_dict["loss"] = loss
                            else:
                                raise ValueError("loss is None")
                        else:
                            raise ValueError("eval_func error")
                    return res_dict
        except:
            raise ValueError("metric_func error")


if __name__ == "__main__":
    # 生成一个虚假的test_df，其中包含user_id, item_id, label，pre_col五列
    import pandas as pd
    import numpy as np

    test_df = pd.DataFrame()
    # user_id定义3个用户，每个用户对应十个item_id
    test_df['user_id'] = [1] * 10 + [2] * 10 + [3] * 10
    test_df['item_id'] = np.random.randint(1, 100, 30)
    #  pre_col随机生成
    test_df['pre_col'] = np.random.rand(30)
    # 为每个用户的每个item_id生成一个ranking
    test_df['ranking'] = test_df.groupby('user_id')['pre_col'].rank(ascending=False, method='first')
    # 按照user_id和ranking排序
    test_df = test_df.sort_values(by=['user_id', 'ranking'])
    # 生成label列
    test_df['label'] = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]
    # 去除ranking列
    test_df = test_df.drop(columns=['ranking'])
    # 默认的col_name
    col_name = {
        'user_col': 'user_id',
        'item_col': 'item_id',
        'label_col': 'label',
        'pre_col': 'pre_col',
    }

    E = EvaluationManager()
    print(E.ndcg(test_df, col_name, k=6, version=1))
