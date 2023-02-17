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

from .config import (
    DefaultsConfig,
    PathConfig,
    RepoConfig,
    ComponentConfig,
)

# def load_environment_from_file(filename):
#     """Utility to initialize a mlonmcu environment from a YAML file."""
#     if isinstance(filename, str):
#         filename = pathlib.Path(filename)
#     with open(filename, encoding="utf-8") as yaml_file:
#         data = yaml.safe_load(yaml_file)
#         if data:
#             if "home" in data:
#                 print(data["home"], filename.parent)
#                 assert os.path.realpath(data["home"]) == os.path.realpath(filename.parent)
#             else:
#                 data["home"] = filename.parent
#             env = Environment(data)
#             return env
#         raise RuntimeError(f"Error opening environment file: {filename}")
#     return None


def load_environment_from_file(filename, base):
    """Utility to initialize a mlonmcu environment from a YAML file."""
    if isinstance(filename, str):
        filename = pathlib.Path(filename)
    with open(filename, encoding="utf-8") as yaml_file:
        loaded = yaml.safe_load(yaml_file)
        if not loaded:
            raise RuntimeError("Invalid YAML contents")
        if "home" in loaded:
            home = loaded["home"]
        else:
            home = None
        if "logging" in loaded:
            if "level" in loaded["logging"]:
                log_level = logging.getLevelName(loaded["logging"]["level"])
            else:
                log_level = None
            if "to_file" in loaded["logging"]:
                log_to_file = bool(loaded["logging"]["to_file"])
            else:
                log_to_file = None
            if "rotate" in loaded["logging"]:
                log_rotate = bool(loaded["logging"]["rotate"])
            else:
                log_rotate = None
        else:
            log_level = None
            log_to_file = False
            log_rotate = False
        if "cleanup" in loaded:
            if "auto" in loaded["cleanup"]:
                cleanup_auto = bool(loaded["cleanup"]["auto"])
            else:
                cleanup_auto = False
            if "auto" in loaded["cleanup"]:
                cleanup_keep = int(loaded["cleanup"]["keep"])
            else:
                cleanup_keep = 100
        else:
            cleanup_auto = False
            cleanup_keep = 100
        if "paths" in loaded:
            paths = {}
            for key in loaded["paths"]:
                path = loaded["paths"][key]
                if isinstance(path, list):
                    paths[key] = []
                    for p in path:
                        paths[key].append(PathConfig(p, base=home))
                else:
                    paths[key] = PathConfig(path, base=home)
        else:
            paths = None
        if "repos" in loaded:
            repos = {}
            for key in loaded["repos"]:
                repo = loaded["repos"][key]
                if "url" not in repo:
                    raise RuntimeError("Missing field 'url' in YAML file")
                if "ref" in repo:
                    repos[key] = RepoConfig(repo["url"], ref=repo["ref"])
                else:
                    repos[key] = RepoConfig(repo["url"])
        else:
            repos = None

        def helper(data):
            if data is None:
                return None
            supported = data.get("supported", [])
            used = data.get("use", [])
            combined = list(set(supported + used))
            return {key: ComponentConfig(key in supported, key in used) for key in combined}
        frameworks = helper(loaded.get("frameworks", None))
        backends = helper(loaded.get("backends", None))
        frontends = helper(loaded.get("frontends", None))
        features = helper(loaded.get("features", None))
        platforms = helper(loaded.get("platforms", None))
        targets = helper(loaded.get("targets", None))
        postprocesses = helper(loaded.get("postprocesses", None))
        toolchains = helper(loaded.get("toolchains", None))
        if "vars" in loaded:
            variables = loaded["vars"]
        else:
            variables = None
        if "flags" in loaded:
            default_flags = loaded["flags"]
        else:
            default_flags = None
        defaults = DefaultsConfig(
            log_level=log_level,
            log_to_file=log_to_file,
            log_rotate=log_rotate,
            cleanup_auto=cleanup_auto,
            cleanup_keep=cleanup_keep,
        )
        env = base(
            home,
            defaults=defaults,
            paths=paths,
            repos=repos,
            frameworks=frameworks,
            backends=backends,
            frontends=frontends,
            platforms=platforms,
            toolchains=toolchains,
            targets=targets,
            features=features,
            postprocesses=postprocesses,
            variables=variables,
            default_flags=default_flags,
        )
        return env
