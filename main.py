'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-22
Description  : 
'''
import subprocess
from managers.logger_manager import logger
import traceback
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

models = ["din"]
for model in models:
    try:
        logger.info(f"开始运行{model}模型")
        logger.send_message(f"开始运行{model}模型", message_type=2)
        subprocess.run(f"python -m models.{model}.run_expid", shell=True)
        logger.send_message(f"运行{model}模型成功", message_type=2, mention=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"运行{model}失败，错误信息如下：{e}")
        continue
    except Exception as e:
        logger.error(f"运行{model}失败，错误信息如下：{traceback.format_exc()}")
        continue

# results_path = get_file_path(['results'])
# subprocess.run(f"tensorboard --logdir {results_path} --load_fast=false", shell=True)
