'''
Author       : wyx-hhhh
Date         : 2024-04-08
LastEditTime : 2024-04-08
Description  : 
'''
import pandas as pd

from utils.file_utils import get_file_path


def split_gowalla_data():
    data_path = get_file_path(["data", "gowalla", "loc-gowalla_totalCheckins.txt"])
    data = pd.read_csv(data_path, sep='\t', header=None, names=["user_id", "check_in_time", "latitude", "longitude", "item_id"])
    data = data[["user_id", "item_id"]]
    # 去除重复数据，去除空数据
    data = data.drop_duplicates()
    data = data.dropna()
    # 去除交互次数小于10的用户
    user_count = data["user_id"].value_counts()
    user_count = user_count[user_count >= 11]
    data = data[data["user_id"].isin(user_count.index)]

    data.to_csv(get_file_path(["data", "gowalla", "gowalla.csv"]), index=False)
    # 数据集处理，分割成训练集、测试集，比例为8:2，并进行存储
    train_data = data.sample(frac=0.8, random_state=2024)[["user_id", "item_id"]]
    test_data = data.drop(train_data.index)[["user_id", "item_id"]]
    train_data.to_csv(get_file_path(["data", "gowalla", "train.csv"]), index=False)
    test_data.to_csv(get_file_path(["data", "gowalla", "test.csv"]), index=False)


def get_gowalla_num():
    data_path = get_file_path(["data", "gowalla", "gowalla.csv"])
    data = pd.read_csv(data_path, usecols=["user_id", "item_id"])
    user_num = data["user_id"].nunique()
    item_num = data["item_id"].nunique()
    return user_num, item_num


if __name__ == "__main__":
    split_gowalla_data()
    print(get_gowalla_num())
