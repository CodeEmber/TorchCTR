'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-09-04
Description  : 
'''
from data.process_data import ProcessData


class DataManager():

    def __init__(self, config: dict) -> None:
        self.config = config
        self.data = self.get_data()

    def get_data(self):
        if self.config["data"] in ["amazon_book", "gowalla", "yelp2018", "alibaba_ifashion", "amazon_kindle", "douban_book"]:
            data = ProcessData(config=self.config)
        else:
            raise ValueError("数据集错误")
        return data

    def data_process(self):
        return self.data.data_process()
