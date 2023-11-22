'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2023-11-22
Description  : 
'''
import subprocess
from utils.file_utils import get_file_path
from utils.logger import MyLogger

logger = MyLogger()

models = ["nfm"]
for model in models:
    try:
        logger.info(f"开始运行{model}模型")
        subprocess.run(f"python -m models.{model}.run_expid.py", shell=True)
    except Exception as e:
        logger.error(f"运行{model}失败")
        logger.error(e)
        continue

# results_path = get_file_path(['results'])
# subprocess.run(f"tensorboard --logdir {results_path} --load_fast=false", shell=True)
