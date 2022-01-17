import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]::%(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

initialized = False


def get_formatter(minimal=False):
    if minimal:
        fmt = "%(levelname)s - %(message)s"
    else:
        fmt = "[%(asctime)s]::%(pathname)s:%(lineno)d::%(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    return formatter


def get_logger():
    global initialized
    logger = logging.getLogger("mlonmcu")
    if len(logger.handlers) == 0:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(get_formatter(minimal=True))
        # stream_handler.setLevel(?)
        logger.addHandler(stream_handler)
        logger.propagate = False
        initialized = True
    return logger


def set_log_level(level):  # command line only!
    logger = logging.getLogger("mlonmcu")
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


def set_log_file(path, level=logging.DEBUG, rotate=False):
    logger = logging.getLogger("mlonmcu")
    if rotate:
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=path, when="midnight", backupCount=30
        )
    else:
        file_handler = logging.FileHandler(path, "w")
    file_handler.setFormatter(get_formatter())
    file_handler.setLevel(level)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(hdlr)
    logger.addHandler(file_handler)
