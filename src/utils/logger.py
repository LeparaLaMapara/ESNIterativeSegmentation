import os
import sys
import logging
import shutil
from datetime import datetime



def get_logger(run_path):
    """Return a console + file logger."""
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',"%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    console_handler= logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    log_dir_path  = os.path.join(run_path,'logs')
    os.makedirs(log_dir_path, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir_path, f"run.log"), mode='w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger