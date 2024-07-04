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

# TODO: rename to paths.py or user.py?

import os
import xdg
import logging
from enum import Enum
from pathlib import Path


def get_config_dir():
    config_dir = os.path.join(xdg.xdg_config_home(), "mlonmcu")
    return config_dir


def init_config_dir():
    config_dir = Path(get_config_dir())
    config_dir.mkdir(parents=True, exist_ok=True)
    subdirs = ["environments", "models", "plugins"]
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


def get_plugins_dir():
    environments_dir = os.path.join(get_config_dir(), "plugins")
    return environments_dir


DEFAULTS = {
    "environment": "default",
    "template": "default",
}

env_subdirs = ["deps", "plugins"]


# class LogLevel(Enum):
#     DEBUG = 0
#     VERBOSE = 1
#     INFO = 2
#     WARNING = 3
#     ERROR = 4


class BaseConfig:
    def __repr__(self):
        return self.__class__.__name__ + "(" + str(vars(self)) + ")"


class DefaultsConfig(BaseConfig):
    # TODO: loglevels enum

    def __init__(
        self,
        log_level=logging.INFO,
        log_to_file=False,
        log_rotate=False,
        default_framework=None,
        default_backends={},
        default_target=None,
        cleanup_auto=False,
        cleanup_keep=100,
    ):
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_rotate = log_rotate
        self.default_framework = default_framework
        self.default_backends = default_backends
        self.default_target = default_target
        self.cleanup_auto = cleanup_auto
        self.cleanup_keep = cleanup_keep


class PathConfig(BaseConfig):
    def __init__(self, path, base=None):
        if isinstance(path, str):
            self.path = Path(path)
        else:
            self.path = path
        if isinstance(base, str):
            self.base = Path(base)
        else:
            self.base = base
        if base:
            if not self.path.is_absolute():
                assert base is not None
                self.path = self.base / self.path
        # Resolve symlinks
        self.path = self.path.resolve()

    def __repr(self):
        return f"PathConfig({self.path})"


class RepoConfig(BaseConfig):
    def __init__(self, url, ref=None, options=None):
        self.url = url
        if ref is not None:
            assert isinstance(ref, str)
        self.ref = ref
        self.options = options if options is not None else {}
        assert isinstance(self.options, dict)

    @property
    def single_branch(self):
        value = self.options.get("single_branch", False)
        assert isinstance(value, bool)
        return value

    @property
    def recursive(self):
        value = self.options.get("recursive", True)
        assert isinstance(value, bool)
        return value

    @property
    def submodules(self):
        value = self.options.get("submodules", None)
        if value is not None:
            assert isinstance(value, list)
        return value


class BackendConfig(BaseConfig):
    def __init__(self, name, enabled=True, features={}):
        self.name = name
        self.enabled = enabled
        self.features = features


class FeatureKind(Enum):
    UNKNOWN = 0
    FRAMEWORK = 1
    BACKEND = 2
    TARGET = 3
    FRONTEND = 4


class FeatureConfig:
    def __init__(self, name, kind=FeatureKind.UNKNOWN, supported=True):
        self.name = name
        self.supported = supported

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(vars(self)) + ")"


class FrameworkFeatureConfig(FeatureConfig):
    def __init__(self, name, framework, supported=True):
        super().__init__(name=name, kind=FeatureKind.FRONTEND, supported=supported)
        self.framework = framework


class BackendFeatureConfig(FeatureConfig):
    def __init__(self, name, backend, supported=True):
        super().__init__(name=name, kind=FeatureKind.FRONTEND, supported=supported)
        self.backend = backend


class PlatformFeatureConfig(FeatureConfig):
    def __init__(self, name, platform, supported=True):
        super().__init__(name=name, kind=FeatureKind.TARGET, supported=supported)
        self.platform = platform


class TargetFeatureConfig(FeatureConfig):
    def __init__(self, name, target, supported=True):
        super().__init__(name=name, kind=FeatureKind.TARGET, supported=supported)
        self.target = target


class FrontendFeatureConfig(FeatureConfig):
    def __init__(self, name, frontend, supported=True):
        super().__init__(name=name, kind=FeatureKind.FRONTEND, supported=supported)
        self.frontend = frontend


class FrameworkConfig(BaseConfig):
    def __init__(self, name, enabled=True, backends={}, features={}):
        self.name = name
        self.enabled = enabled
        self.backends = backends
        self.features = features


class FrontendConfig(BaseConfig):
    def __init__(self, name, enabled=True, features={}):
        self.name = name
        self.enabled = enabled
        self.features = features


class PlatformConfig(BaseConfig):
    def __init__(self, name, enabled=True, features={}):
        self.name = name
        self.enabled = enabled
        self.features = features


class TargetConfig(BaseConfig):
    def __init__(self, name, enabled=True, features={}):
        self.name = name
        self.enabled = enabled
        self.features = features
