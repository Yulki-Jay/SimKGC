import logging


def _setup_logger(): # 这个函数的作用是设置日志的格式
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger() # 这个logger是全局的，可以在任何地方使用，实现日志信息在整个项目中的一致性记录
