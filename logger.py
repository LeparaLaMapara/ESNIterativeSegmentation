import os
import sys
import logging
import shutil
from datetime import datetime


def get_logger(filepath=None):
    """Return a console + file logger."""
    log = logging.getLogger('Video')

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    log_format = logging.Formatter(
        fmt='[%(asctime)s][%(filename)s:%(lineno)d] %(message)s'
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_format)
    log.addHandler(console_handler)

    if filepath is not None:
        file_handler = logging.FileHandler(filepath, mode='a')
        file_handler.setFormatter(log_format)
        log.addHandler(file_handler)

    log.setLevel(logging.INFO)

    return log