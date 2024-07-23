'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-07-18
Description  : 
'''
import time
from managers.logger_manager import logger


def time_middleware(custom_text=None):

    def decorator(func):

        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            if custom_text:
                logger.info(custom_text + f"，Function '{func.__name__}' took {execution_time:.4f} seconds to execute.")
            else:
                logger.info(f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute.")
            return result

        return wrapper

    return decorator
