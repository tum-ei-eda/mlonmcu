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
        path: (
            str(path_config.path)
            if isinstance(path_config, PathConfig)
            else [str(config.path) for config in path_config]
        )
        for path, path_config in environment.paths.items()
    }  # TODO: allow relative paths
    data["repos"] = {repo: vars(repo_config) for repo, repo_config in environment.repos.items()}
    # TODO: test with options!
    data["frameworks"] = {
        "default": environment.defaults.default_framework if environment.defaults.default_framework else None,
        **{
            framework.name: {
                "enabled": framework.enabled,
                "backends": {
                    "default": (
                        environment.defaults.default_backends[framework.name]
                        if environment.defaults.default_backends
                        and environment.defaults.default_backends[framework.name]
                        else None
                    ),
                    **{
                        backend.name: {
                            "enabled": backend.enabled,
                            "features": {
                                backend_feature.name: backend_feature.supported for backend_feature in backend.features
                            },
                        }
                        for backend in framework.backends
                    },
                },
                "features": {
                    framework_feature.name: framework_feature.supported for framework_feature in framework.features
                },
            }
            for framework in environment.frameworks
        },
    }
    data["frontends"] = {
        # "default": None,  # unimplemented?
        **{
            frontend.name: {
                "enabled": frontend.enabled,
                "features": {
                    frontend_feature.name: frontend_feature.supported for frontend_feature in frontend.features
                },
            }
            for frontend in environment.frontends
        },
    }
    data["platforms"] = {
        # "default": None,  # unimplemented?
        **{
            platform.name: {
                "enabled": platform.enabled,
                "features": {
                    platform_feature.name: platform_feature.supported for platform_feature in platform.features
                },
            }
            for platform in environment.platforms
        },
    }
    data["toolchains"] = environment.toolchains
    data["targets"] = {
        "default": environment.defaults.default_target,
        **{
            target.name: {
                "enabled": target.enabled,
                "features": {target_feature.name: target_feature.supported for target_feature in target.features},
            }
            for target in environment.targets
        },
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
