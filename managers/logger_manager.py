'''
Description  : 日志文件配置
Author       : wyx-hhhh
Date         : 2022-01-24 17:26:50
LastEditTime : 2024-03-25
LastEditors  : Please set LastEditors
'''
import logging.config
from types import TracebackType
import requests
import json
from config.global_config import GOLBAL_CONFIG
from utils.file_utils import get_file_path
from utils.utilities import format_time
import inspect
from logging import Logger


class LoggerManager():

    def __init__(self, config) -> None:
        self.config = config
        logger_path = get_file_path(path=["config", "logging.ini"])
        logging.config.fileConfig(logger_path)
        self.logger = logging.getLogger(GOLBAL_CONFIG.get('logger_level', 'test'))

    def log(self, level, message):
        frame_info = inspect.stack()[2]
        file_name = frame_info.filename
        file_name = file_name.split('/')[-2] + '/' + file_name.split('/')[-1]
        line_number = frame_info.lineno
        func_name = frame_info.function
        log_message = f"[{file_name}:{line_number}:{func_name}] {message} "
        self.logger.log(logging.getLevelName(level.upper()), log_message)

    def info(self, message):
        self.log('info', message)

    def debug(self, message):
        self.log('debug', message)

    def warning(self, message):
        self.log('warning', message)

    def error(self, message):
        self.log('error', message)
        self.send_message(message, message_type=1, mention=True)

    def _format_dict_message(self, message, color):
        if 'time' in message:
            message['time'] = format_time(message['time'])

        attachments = [{
            "color": color,
            "title": "Model Information",
            "fields": [{
                "title": k,
                "value": json.dumps(v) if isinstance(v, dict) else str(v),
                "short": False if k in ['train', 'valid', 'test'] else True
            } for k, v in message.items()],
        }]
        message = {"attachments": attachments}
        return message

    def _format_str_message(self, message, color):
        message = {
            "attachments": [
                {
                    "color": color,
                    "text": str(message)
                },
            ]
        }
        return message

    def send_message(self, message, message_type=0, message_content_type=1, mention=False):
        """发送消息到slack

        Args:
            message (dict｜str): 消息内容
            message_type (int, optional): 消息类型. Defaults to 0. 0: info, 1: error, 2: success
            message_content_type (int, optional): 消息内容类型. Defaults to 0. 0: dict, 1: str
            mention (bool, optional): 是否@自己. Defaults to False.

        Raises:
            ValueError: 只支持dict和str类型的消息内容, message_type只支持0, 1, 2
        """
        if self.config.get("is_slack_enabled") and self.config.get("slack_url") and self.config.get("slack_user_id"):
            headers = {"Content-type": "application/json"}
            if message_type == 0:
                color = "#ecd452"
            elif message_type == 1:
                color = "#ff0000"
            elif message_type == 2:
                color = "#36a64f"
            else:
                raise ValueError("message_type只支持0, 1, 2，分别代表info, error, success")

            if message_content_type == 0:
                message = self._format_dict_message(message, color)
            elif message_content_type == 1:
                message = self._format_str_message(message, color)
            else:
                raise ValueError("message_content_type只支持0, 1，分别代表dict, str")

            if mention:
                message_text = {"text": f"<@{self.config.get('slack_user_id')}>", "attachments": message["attachments"]}
            else:
                message_text = message

            requests.post(self.config.get("slack_url"), headers=headers, json=message_text)


logger = LoggerManager(config=GOLBAL_CONFIG)

# if __name__ == '__main__':
# evaluation_results = {"epoch": 5, "model_name": "ncf", "train": {"hitrate": 0.1, "ndcg": 0.2}, "valid": {"hitrate": 0.3, "ndcg": 0.4}}
# logger.send_message(evaluation_results, message_type=2, message_content_type=0, mention=True)
