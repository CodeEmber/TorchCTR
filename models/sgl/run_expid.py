'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-09-05
Description  : 
'''
import argparse

from models.sgl.model import SGL
from models.sgl.train_config import train_config

parser = argparse.ArgumentParser(description='运行模型')
parser.add_argument('--data', type=str, default='', help='选择数据集')

# 如果存在data则使用传入的data，否则使用默认的data
train_config['data'] = parser.parse_args().data if parser.parse_args().data else train_config['data']
SGL(train_config).run()
