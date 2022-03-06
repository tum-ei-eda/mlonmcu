#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Loging utilities for MLonMCU"""

import logging
import logging.handlers
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]::%(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

initialized = False


def get_formatter(minimal=False):
    """Returns a log formatter for one on two predefined formats."""
    if minimal:
        fmt = "%(levelname)s - %(message)s"
    else:
        fmt = "[%(asctime)s]::%(pathname)s:%(lineno)d::%(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    return formatter


def get_logger():
    """Helper function which return the main mlonmcu logger while ensuring that is is properly initialized."""
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


def set_log_level(level):
    """Set command line log level at runtime."""
    logger = logging.getLogger("mlonmcu")
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


def set_log_file(path, level=logging.DEBUG, rotate=False):
    """Enable logging to a file."""
    logger = logging.getLogger("mlonmcu")
    if rotate:
        file_handler = logging.handlers.TimedRotatingFileHandler(filename=path, when="midnight", backupCount=30)
    else:
        file_handler = logging.FileHandler(path, mode="a")
    file_handler.setFormatter(get_formatter())
    file_handler.setLevel(level)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    logger.addHandler(file_handler)
