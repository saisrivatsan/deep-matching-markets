import os
import logging

""" Loggers """
def init_logger(log_fname, filemode = 'w'):
    
    root_dir = os.path.dirname(log_fname)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler(log_fname, filemode)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger