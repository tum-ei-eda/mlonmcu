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
""" MLonMCU Features API"""

from abc import ABC
from typing import List

from mlonmcu.config import filter_config
from .type import FeatureType


# TODO: features might get an optional context parameter to lookup if they are supported by themselfs in the environment


class FeatureBase(ABC):
    """Feature base class"""

    feature_type = None

    DEFAULTS = {"enabled": True}
    REQUIRED = []

    def __init__(self, name, config=None):
        self.name = name
        self.config = config if config else {}
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.REQUIRED)

    @property
    def enabled(self):
        return bool(self.config.get("enabled", None))

    def remove_config_prefix(self, config):  # TODO: move to different place
        def helper(key):
            return key.split(f"{self.name}.")[-1]

        return {helper(key): value for key, value in config.items() if f"{self.name}." in key}

    def __repr__(self):
        return type(self).__name__ + f"({self.name})"

    # @property
    # def types(self):
    #     return [base.feature_type for base in type(self).__bases__]

    @classmethod
    def types(cls):
        """Find out which types the features is based on."""
        return [base.feature_type for base in cls.__bases__]

    # This does not make sense because the get_?_config methods may beed a parameter
    # This could be solved by seeting he backend/target/frontend in the constructor!
    # Multiple inheritance would make this still pretty dirty
    # def get_config(self):
    #     for feature_type in self.types:
    #         type_name = FeatureType(feature_type).name.lower()
    #         method_name = f"get_{type_name}_config"
    #         method = getattr(self, method_name)
    #         args = {"type_name": getattr(self, type_name)}
    #         self.method(**args)


class Feature(FeatureBase):
    """Feature of unknown type"""

    feature_type = FeatureType.OTHER


class FrontendFeature(FeatureBase):
    """Frontend related feature"""

    feature_type = FeatureType.FRONTEND

    def __init__(self, name, config=None):
        super().__init__(name, config=config)

    def get_frontend_config(self, frontend):
        # pylint: disable=unused-argument
        return {}

    def add_frontend_config(self, frontend, config):
        config.update(self.get_frontend_config(frontend))


class FrameworkFeature(FeatureBase):
    """Framework related feature"""

    feature_type = FeatureType.FRAMEWORK

    def __init__(self, name, config=None):
        super().__init__(name, config=config)

    def get_framework_config(self, framework):
        # pylint: disable=unused-argument
        return {}

    def add_framework_config(self, framework, config):
        config.update(self.get_framework_config(framework))


class BackendFeature(FeatureBase):
    """Backend related feature"""

    feature_type = FeatureType.BACKEND

    def __init__(self, name, config=None):
        super().__init__(name, config=config)

    def get_backend_config(self, backend):
        # pylint: disable=unused-argument
        return {}

    def add_backend_config(self, backend, config):
        # TODO: cfg passed to method instead of contructor or self.config = config
        config.update(self.get_backend_config(backend))


class TargetFeature(FeatureBase):
    """Target related feature"""

    feature_type = FeatureType.TARGET

    def __init__(self, name, config=None):
        super().__init__(name, config=config)

    def get_target_config(self, target):
        # pylint: disable=unused-argument
        return {}

    def add_target_config(self, target, config):
        # TODO: cfg passed to method instead of contructor or self.config = config
        config.update(self.get_target_config(target))


class PlatformFeature(FeatureBase):
    """Platform/Compile related feature"""

    feature_type = FeatureType.PLATFORM

    def __init__(self, name, config=None):
        super().__init__(name, config=config)

    def get_platform_config(self, platform):
        return {}

    def add_platform_config(self, platform, config):
        config.update(self.get_platform_config(platform))

    def get_cmake_args(self):
        return []

    def add_cmake_args(self, args):
        args += self.get_cmake_args()

    # TODO: alternative mlif.cmake_args appenden?


class SetupFeature(FeatureBase):  # TODO: alternative: CacheFeature
    """Setup/Cache related feature"""

    feature_type = FeatureType.SETUP

    def __init__(self, name, config=None):
        super().__init__(name, config=config)

    def get_setup_config(self):
        raise NotImplementedError
        return {}

    def add_setup_config(self, config):
        raise NotImplementedError
        config.update(self.get_setup_config(compile))

    def get_required_cache_flags(self):
        return {}

    def add_required_cache_flags(self, required_flags):
        own_flags = self.get_required_cache_flags()
        for key, flags in own_flags.items():
            if key in required_flags:
                required_flags[key].append(flags)
            else:
                required_flags[key] = flags


class RunFeature(FeatureBase):
    """Run related feature"""

    feature_type = FeatureType.RUN

    def __init__(self, name, config=None):
        super().__init__(name, config=config)

    def get_run_config(self):
        return {}

    def add_run_config(self, config):
        config.update(self.get_run_config())
