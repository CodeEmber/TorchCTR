'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-07-22
Description  : 
'''
import subprocess
import time
from managers.logger_manager import logger
import traceback
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

models = ["lightgcn_matrix"]
for model in models:
    try:
        start_time = time.time()
        logger.info(f"开始运行{model}模型，开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        logger.send_message(f"开始运行{model}模型，开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}", message_type=2)
        subprocess.run(f"python -m models.{model}.run_expid", shell=True, check=True, capture_output=True)
        end_time = time.time()
        run_time = end_time - start_time
        hours, rem = divmod(run_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f"运行{model}模型成功，耗时{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
        logger.send_message("运行{}模型成功，耗时{:0>2}:{:0>2}:{:05.2f}".format(model, int(hours), int(minutes), seconds), message_type=2, mention=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"运行{model}失败", e)
        continue
    except Exception as e:
        logger.error(f"运行{model}失败", e)
        continue

# results_path = get_file_path(['results'])
# subprocess.run(f"tensorboard --logdir {results_path} --load_fast=false", shell=True)
