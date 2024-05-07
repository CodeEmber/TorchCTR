'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-04-25
Description  : 
'''
import pandas as pd

from data.criteo.process_data import CriteoProcessData
from data.movielens.process_data import MovieLenProcessData
from data.amazon.process_data import AmazonProcessData
from data.gowalla.process_data import GowallaProcessData
from data.pixelrec.process_data import PixelRecProcessData


class DataManager():

    def __init__(self, config: dict) -> None:
        self.config = config
        self.data = self.get_data()

    def get_data(self):
        if self.config["data"] == "criteo":
            data = CriteoProcessData(config=self.config)
        elif self.config["data"] == "movielens":
            data = MovieLenProcessData(config=self.config)
        elif self.config["data"] == "amazon":
            data = AmazonProcessData(config=self.config)
        elif self.config["data"] == "gowalla":
            data = GowallaProcessData(config=self.config)
        elif self.config["data"] == "pixelrec":
            data = PixelRecProcessData(config=self.config)
        else:
            raise ValueError("数据集错误")
        return data

    def data_process(self):
        return self.data.data_process()
