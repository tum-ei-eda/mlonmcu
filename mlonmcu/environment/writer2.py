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
import yaml
import pathlib
import logging

from .config import PathConfig


def create_environment_dict(environment):
    data = {}
    data["home"] = environment.home
    data["logging"] = {
        "level": logging.getLevelName(environment.defaults.log_level),
        "to_file": environment.defaults.log_to_file,
        "rotate": environment.defaults.log_rotate,
    }
    data["cleanup"] = {
        "auto": environment.defaults.cleanup_auto,
        "keep": environment.defaults.cleanup_keep,
    }
    data["paths"] = {
        path: str(path_config.path)
        if isinstance(path_config, PathConfig)
        else [str(config.path) for config in path_config]
        for path, path_config in environment.paths.items()
    }  # TODO: allow relative paths
    data["repos"] = {repo: vars(repo_config) for repo, repo_config in environment.repos.items()}
    data["frameworks"] = {
        "supported": [name for name, value in environment.frameworks.items() if value.supported],
        "use": [name for name, value in environment.frameworks.items() if value.used],
    }
    data["backends"] = {
        "supported": [name for name, value in environment.backends.items() if value.supported],
        "use": [name for name, value in environment.backends.items() if value.used],
    }
    data["frontends"] = {
        "supported": [name for name, value in environment.frontends.items() if value.supported],
        "use": [name for name, value in environment.frontends.items() if value.used],
    }
    data["toolchains"] = {
        "supported": [name for name, value in environment.toolchains.items() if value.supported],
        "use": [name for name, value in environment.toolchains.items() if value.used],
    }
    data["platforms"] = {
        "supported": [name for name, value in environment.platforms.items() if value.supported],
        "use": [name for name, value in environment.platforms.items() if value.used],
    }
    data["targets"] = {
        "supported": [name for name, value in environment.targets.items() if value.supported],
        "use": [name for name, value in environment.targets.items() if value.used],
    }
    data["features"] = {
        "supported": [name for name, value in environment.features.items() if value.supported],
        "use": [name for name, value in environment.features.items() if value.used],
    }
    data["postprocesses"] = {
        "supported": [name for name, value in environment.postprocesses.items() if value.supported],
        "use": [name for name, value in environment.postprocesses.items() if value.used],
    }
    data["vars"] = environment.vars
    data["flags"] = environment.flags
    return data


def write_environment_to_file(environment, filename):
    """Utility to initialize a mlonmcu environment from a YAML file."""
    if isinstance(filename, str):
        filename = pathlib.Path(filename)
    data = create_environment_dict(environment)
    with open(filename, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)
