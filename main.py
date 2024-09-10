'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-09-10
Description  : 
'''
import subprocess
import time
from managers.logger_manager import logger
import traceback
import os
import tensorboard

os.environ["MKL_THREADING_LAYER"] = "GNU"

# run_list = ["lightgcn|gowalla", "lightgcn|amazon_book", "lightgcn|alibaba_ifashion", "lightgcn|amazon_kindle"]
# run_list = ["ngcf|gowalla", "ngcf|yelp2018", "ngcf|amazon_book", "ngcf|alibaba_ifashion", "ngcf|amazon_kindle"]
# run_list = ["sgl|gowalla", "sgl|yelp2018", "sgl|amazon_book", "sgl|alibaba_ifashion", "sgl|amazon_kindle"]
# run_list = ["simgcl|amazon_kindle", "simgcl|yelp2018", "simgcl|amazon_book", "simgcl|alibaba_ifashion"]
# run_list = ["xsimgcl|gowalla", "xsimgcl|amazon_kindle", "xsimgcl|yelp2018", "xsimgcl|amazon_book", "xsimgcl|alibaba_ifashion"]
# run_list = ["ngcf|gowalla", "ngcf|yelp2018", "ngcf|amazon_book"]
# run_list = ["sgl|gowalla", "sgl|yelp2018", "sgl|amazon_book"]
# run_list = ["simgcl|amazon_book", "simgcl|yelp2018", "simgcl|gowalla"]
# run_list = ["lightgcn|yelp2018"]
run_list = ["lightgcn|amazon_book"]
# run_list = ["ngcf|gowalla"]
# run_list = ["ngcf|yelp2018"]
# run_list = ["ngcf|amazon_book"]
# run_list = ["sgl|yelp2018"]
# run_list = ["sgl|gowalla"]
# run_list = ["sgl|alibaba_ifashion"]
# run_list = ["sgl|amazon_book"]
# run_list = ["simgcl|yelp2018"]
# run_list = ["simgcl|gowalla"]
# run_list = ["simgcl|amazon_book"]
# run_list = ["xsimgcl|yelp2018"]
# run_list = ["xsimgcl|gowalla"]
# run_list = ["xsimgcl|amazon_kindle"]
# run_list = ["xsimgcl|alibaba_ifashion"]
for run in run_list:
    try:
        start_time = time.time()
        logger.info(f"开始运行{run}模型，开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        logger.send_message(f"开始运行{run}模型，开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}", message_type=2)
        if "|" in run:
            model, data = run.split("|")
        else:
            model = run
            data = ""
        if data == "":
            subprocess.run(f"python -m models.{model}.run_expid", shell=True, check=True, capture_output=True)
        else:
            subprocess.run(f"python -m models.{model}.run_expid --data {data}", shell=True, check=True, capture_output=True)

        end_time = time.time()
        run_time = end_time - start_time
        hours, rem = divmod(run_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f"运行{run}模型成功，耗时{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
        logger.send_message("运行{}模型成功，耗时{:0>2}:{:0>2}:{:05.2f}".format(run, int(hours), int(minutes), seconds), message_type=2, mention=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"运行{run}失败,{traceback.format_exc()}")
        continue
    except Exception as e:
        logger.error(f"运行{run}失败,{traceback.format_exc()}")
        continue

# results_path = get_file_path(['results'])
# subprocess.run(f"tensorboard --logdir /home/wyx/TorchCTR/results --load_fast=false", shell=True)
