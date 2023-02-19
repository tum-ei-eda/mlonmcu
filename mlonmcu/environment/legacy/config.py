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

import logging
from enum import Enum

from ..config import DefaultsConfig, BaseConfig


class DefaultsConfigOld(DefaultsConfig):
    def __init__(
        self,
        log_level=logging.INFO,
        log_to_file=False,
        log_rotate=False,
        cleanup_auto=False,
        cleanup_keep=100,
        default_framework=None,
        default_backends={},
        default_target=None,
    ):
        super().__init__(
            log_level=log_level,
            log_to_file=log_to_file,
            log_rotate=log_rotate,
            cleanup_auto=False,
            cleanup_keep=100,
        )
        self.default_framework = default_framework
        self.default_backends = default_backends
        self.default_target = default_target


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
