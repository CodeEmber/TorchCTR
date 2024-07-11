'''
Author       : wyx-hhhh
Date         : 2023-10-28
LastEditTime : 2024-03-25
Description  : 
'''
import time


def time_middleware(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute.")
        return result

    return wrapper
