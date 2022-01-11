# TODO: rename to paths.py or user.py?

import os
import xdg
from pathlib import Path


def get_config_dir():
    config_dir = os.path.join(xdg.xdg_config_home(), "mlonmcu")
    return config_dir


def init_config_dir():
    config_dir = Path(get_config_dir())
    config_dir.mkdir()
    subdirs = ["environments", "models"]
    files = ["environments.ini"]
    for subdir in subdirs:
        environments_dir = config_dir / subdir
        environments_dir.mkdir(exist_ok=True)
    for file in files:
        filepath = config_dir / file
        filepath.touch(exist_ok=True)


def get_environments_dir():
    environments_dir = os.path.join(get_config_dir(), "environments")
    return environments_dir


def get_environments_file():
    environments_file = os.path.join(get_config_dir(), "environments.ini")
    return environments_file


DEFAULTS = {
    "environment": "default",
    "template": "default",
}

env_subdirs = ["deps"]
