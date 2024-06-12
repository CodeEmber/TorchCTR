'''
Author       : wyx-hhhh
Date         : 2024-05-29
LastEditTime : 2024-05-29
Description  : 
'''
import time
import pynvml

from managers.logger_manager import LoggerManager


class GPUMonitor():

    def __init__(self, config: dict, logger: LoggerManager) -> None:
        self.config = config
        self.logger = logger
        try:
            pynvml.nvmlInit()
            self.driver_version = pynvml.nvmlSystemGetDriverVersion()
            self.cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            self.num_device = pynvml.nvmlDeviceGetCount()
            self.gpu_info = {
                "driver_version": "None",
                "cuda_version": "None",
                "num_device": "None",
                "device": [],
            }
            self.run()
        except:
            self.config["device"] = -1
            self.logger.info("未检测到GPU设备")
            self.logger.send_message("未检测到GPU设备，使用CPU", message_type=1, message_content_type=1)

    def get_gpu_info(self):
        self.gpu_info['driver_version'] = self.driver_version
        self.gpu_info['cuda_version'] = self.cuda_version
        self.gpu_info['num_device'] = self.num_device
        for idx in range(self.num_device):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            device_name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem = int(int(mem_info.total) / 1024 / 1024)
            used_mem = int(int(mem_info.used) / 1024 / 1024)
            free_mem = int(int(mem_info.free) / 1024 / 1024)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            self.gpu_info["device"].append({
                "device_name": device_name,
                "idx": idx,
                "temp": temp,
                "used_mem": used_mem,
                "free_mem": free_mem,
                "total_mem": total_mem,
                "gpu_util": util,
            })
            self.gpu_info["device"].sort(key=lambda x: x["free_mem"], reverse=True)

    def run(self):
        cycle_num = self.config.get("cycle_num", 30)
        while cycle_num > 0:
            self.get_gpu_info()
            if self.gpu_info["num_device"] == 0:
                self.config["device"] = -1
                self.logger.info("未检测到GPU设备")
                self.logger.send_message("未检测到GPU设备，使用CPU", message_type=1, message_content_type=1)
                break
            need_free_mem = self.config.get("need_free_mem", 1000)
            if self.gpu_info["device"][0]["free_mem"] > need_free_mem:
                self.config["device"] = self.gpu_info["device"][0]["idx"]

                self.logger.info(f"选择GPU: {self.gpu_info['device'][0]['idx']}, 可用内存: {self.gpu_info['device'][0]['free_mem']}MB")
                self.logger.send_message(f"选择GPU: {self.gpu_info['device'][0]['idx']}, 可用内存: {self.gpu_info['device'][0]['free_mem']}MB", message_type=2, message_content_type=1)
                break
            else:
                self.logger.info("GPU资源不足，等待10分钟后重新检查")
                self.logger.send_message("GPU资源不足，等待10分钟后重新检查", message_type=1, message_content_type=1)
                time.sleep(600)
                cycle_num -= 1
