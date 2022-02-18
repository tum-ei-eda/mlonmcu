import os
import configparser
from datetime import datetime as dt

# TODO: EnvironmentList/Factory/Registry? also via code?
from .config import get_environments_file


def validate_name(name):
    # TODO: regex for valid names without spaces etc
    return True


def get_environments_map():
    result = {}
    environments_file = get_environments_file()
    if os.path.isfile(environments_file):
        parser = configparser.ConfigParser()
        parser.read(environments_file)
        for env_name in parser.sections():
            result[env_name] = parser[env_name]
    return result


def get_environment_names():
    envs_dict = get_environments_map()
    return envs_dict.keys()


def get_alternative_name(name, names):
    current = name
    i = -1
    while current in names:
        i = i + 1
        if i == 0:
            current = current + "_" + str(i)
        else:
            temp = current.split("_")
            current = "_".join(temp[:-1]) + "_" + str(i)
    return current


def register_environment(name, path, overwrite=False):
    validate_name(name)
    if not os.path.isabs(path):
        raise RuntimeError("Not an absolute path")

    environments_file = get_environments_file()
    if not os.path.isfile(environments_file):
        raise RuntimeError("Environments file does not yet exist")

    # with open(environments_file, "a") as envs_file:
    #    envs_file.write(name + "=" + path)
    config = configparser.ConfigParser()
    config.read(environments_file)
    now = dt.now()
    created = now.strftime("%Y%d%mT%H%M%S")
    if name in config.sections() and not overwrite:
        raise RuntimeError(f"Environment with name {name} does already exist")
    config[name] = {"path": path, "created": created}
    environments_file = get_environments_file()
    with open(environments_file, "w") as handle:
        config.write(handle)
