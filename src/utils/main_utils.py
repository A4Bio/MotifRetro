import os
import logging

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def print_log(message):
    print(message)
    logging.info(message)
