'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-08-27
Description  : 
'''
import time


def time_middleware(custom_text=None):

    def decorator(func):

        def wrapper(*args, **kwargs):
            from managers.logger_manager import logger
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            text = f"{custom_text}，Function '{func.__name__}' took {execution_time:.4f} seconds to execute." if custom_text else f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute."
            logger.info(text)
            return result

        return wrapper

    return decorator
