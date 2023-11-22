'''
Description  : 日志文件配置
Author       : wyx-hhhh
Date         : 2022-01-24 17:26:50
LastEditTime : 2023-10-30
LastEditors  : Please set LastEditors
'''
import logging.config

from utils.file_utils import get_file_path


class MyLogger:

    def __new__(cls):
        logger_path = get_file_path(path=["config", "logging.ini"])
        logging.config.fileConfig(logger_path)
        return logging.getLogger('test')


if __name__ == '__main__':
    logger = MyLogger().logger
    logger.info('test')
