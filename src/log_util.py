import logging
import time


# log
def create_logger():
    time_str = str(int(time.time()))[:8]
    logger = logging.getLogger("main")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(f"../logs/log_{time_str}.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger