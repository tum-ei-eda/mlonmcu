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
from abc import ABC

from mlonmcu.feature.type import FeatureType
from mlonmcu.config import filter_config
from mlonmcu.feature.features import get_matching_features


class Framework(ABC):
    registry = {}

    name = None

    FEATURES = set()
    DEFAULTS = {}
    REQUIRED = {"tf.src_dir"}
    OPTIONAL = set()

    def __init__(self, features=None, config=None, backends={}):
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.OPTIONAL, self.REQUIRED)

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.FRAMEWORK)
        for feature in features:
            assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
            feature.used = True
            feature.add_framework_config(self.name, self.config)
        return features

    def remove_config_prefix(self, config):
        def helper(key):
            return key.split(f"{self.name}.")[-1]

        return {helper(key): value for key, value in config if f"{self.name}." in key}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.name, str)
        cls.registry[cls.name] = cls

    def get_platform_config(self, platform):
        return {}

    def add_platform_config(self, platform, config):
        config.update(self.get_platform_config(platform))

    def get_platform_defs(self, platform):
        if platform == "espidf":
            framework_upper = self.name.upper()
            return {f"MLONMCU_FRAMEWORK_{framework_upper}": True}
        else:
            return {"MLONMCU_FRAMEWORK": self.name}

    def add_platform_defs(self, platform, defs):
        defs.update(self.get_platform_defs(platform))
