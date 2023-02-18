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
import logging
import tempfile
from abc import ABC
from pathlib import Path
from typing import Dict, List, Union

from .config import (
    DefaultsConfig,
    PathConfig,
    RepoConfig,
    ComponentConfig,
    # FrameworkConfig,
    # FrameworkFeatureConfig,
    # BackendConfig,
    # BackendFeatureConfig,
    # TargetConfig,
    # TargetFeatureConfig,
    # PlatformConfig,
    # PlatformFeatureConfig,
    # FrontendConfig,
    # FrontendFeatureConfig,
)

# from mlonmcu.feature.type import FeatureType


# def _feature_helper(obj, name):
#     if not obj.enabled:
#         return []
#     features = obj.features
#     if name:
#         return [feature for feature in features if feature.name == name]
#     else:
#         return features


# def _extract_names(objs):
#     return [obj.name for obj in objs]


# def _filter_enabled(objs):
#     return [obj for obj in objs if obj.enabled]


class BaseEnvironment(ABC):
    def __init__(self):
        self.version = None
        self._home: Path = None


class Environment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.alias: str = None
        self.version = 2
        self.defaults = DefaultsConfig()
        self.paths: "Dict[str, Union[PathConfig, List[PathConfig]]]" = {}
        self.repos: "Dict[str, RepoConfig]" = {}
        self.frameworks: "Dict[str, ComponentConfig]" = {}
        self.backends: "Dict[str, ComponentConfig]" = {}
        self.features: "Dict[str, ComponentConfig]" = {}
        self.postprocesses: "Dict[str, ComponentConfig]" = {}
        self.toolchains: "Dict[str, ComponentConfig]" = {}
        self.frontends: "Dict[str, ComponentConfig]" = {}
        self.platforms: "Dict[str, ComponentConfig]" = {}
        self.toolchains: "Dict[str, ComponentConfig]" = {}
        self.targets: "Dict[str, ComponentConfig]" = {}
        self.vars = {}
        self.flags = {}

    def __str__(self):
        return self.__class__.__name__ + "(" + str(vars(self)) + ")"

    @property
    def home(self):
        """Home directory of mlonmcu environment."""
        return self._home

    def lookup_path(self, name):
        assert name in self.paths, f"Unable to find '{name}' path in environment config"
        return self.paths[name]

    def lookup_var(self, name, default=None):
        return self.vars.get(name, default)

    def get_supported_features(self):
        supported = [name for name, config in self.features.items() if config.supported]
        return supported

    def get_used_features(self):
        used = [name for name, config in self.features.items() if config.used]
        return used

    def has_feature(self, name):
        return name in self.get_supported_features()

    def get_supported_frontends(self):
        supported = [name for name, config in self.frontends.items() if config.supported]
        return supported

    def get_used_frontends(self):
        used = [name for name, config in self.frontends.items() if config.used]
        return used

    def has_frontend(self, name):
        return name in self.get_supported_frontends()

    def get_supported_backends(self):
        supported = [name for name, config in self.backends.items() if config.supported]
        return supported

    def get_used_backends(self):
        used = [name for name, config in self.backends.items() if config.used]
        return used

    def has_backend(self, name):
        return name in self.get_supported_backends()

    def get_supported_frameworks(self):
        supported = [name for name, config in self.frameworks.items() if config.supported]
        return supported

    def get_used_frameworks(self):
        used = [name for name, config in self.frameworks.items() if config.used]
        return used

    def has_framework(self, name):
        return name in self.get_supported_frameworks()

    def get_supported_platforms(self):
        supported = [name for name, config in self.platforms.items() if config.supported]
        return supported

    def get_used_platforms(self):
        used = [name for name, config in self.platforms.items() if config.used]
        return used

    def has_platform(self, name):
        return name in self.get_supported_platforms()

    def get_supported_target(self):
        supported = [name for name, config in self.targets.items() if config.supported]
        return supported

    def get_used_targets(self):
        used = [name for name, config in self.targets.items() if config.used]
        return used

    def has_target(self, name):
        return name in self.get_supported_targets()

    def has_toolchain(self, name):
        return self.toolchains.get(name, False)

    # def has_postprocess(self, name):
    #     configs = self.lookup_postprocess_configs(postprocess=name)
    #     return len(configs) > 0

    def get_default_backends(self, framework):
        # TODO: deprecate
        return self.get_used_backends()

    def get_default_frameworks(self):
        # TODO: deprecate
        return self.get_used_frameworks()

    def get_default_targets(self):
        # TODO: deprecate
        return self.get_used_targets()


class DefaultEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.defaults = DefaultsConfig(
            log_level=logging.DEBUG,
            log_to_file=False,
            cleanup_auto=False,
            cleanup_keep=100,
        )
        self.paths = {
            "deps": PathConfig("./deps"),
            "logs": PathConfig("./logs"),
            "results": PathConfig("./results"),
            "plugins": PathConfig("./plugins"),
            "temp": PathConfig("out"),
            "models": [
                PathConfig("./models"),
            ],
        }
        # TODO: refresh or remove
        self.repos = {
            "tensorflow": RepoConfig("https://github.com/tensorflow/tensorflow.git", ref="v2.5.2"),
            "tflite_micro_compiler": RepoConfig(
                "https://github.com/cpetig/tflite_micro_compiler.git", ref="master"
            ),  # TODO: freeze ref?
            "tvm": RepoConfig(
                "https://github.com/tum-ei-eda/tvm.git", ref="tumeda"
            ),  # TODO: use upstream repo with suitable commit?
            "utvm_staticrt_codegen": RepoConfig(
                "https://github.com/tum-ei-eda/utvm_staticrt_codegen.git", ref="master"
            ),  # TODO: freeze ref?
            "etiss": RepoConfig("https://github.com/tum-ei-eda/etiss.git", ref="master"),  # TODO: freeze ref?
        }
        self.frameworks = {}
        self.backends = {}
        self.features = {}
        self.postprocesses = {}
        self.toolchains = {}
        # self.frameworks = [
        #     FrameworkConfig(
        #         "tflm",
        #         enabled=True,
        #         backends=[
        #             BackendConfig("tflmc", enabled=True, features=[]),
        #             BackendConfig("tflmi", enabled=True, features=[]),
        #         ],
        #         features=[
        #             FrameworkFeatureConfig("muriscvnn", framework="tflm", supported=False),
        #         ],
        #     ),
        #     FrameworkConfig(
        #         "utvm",
        #         enabled=True,
        #         backends=[
        #             BackendConfig(
        #                 "tvmaot",
        #                 enabled=True,
        #                 features=[
        #                     BackendFeatureConfig("unpacked_api", backend="tvmaot", supported=True),
        #                 ],
        #             ),
        #             BackendConfig("tvmrt", enabled=True, features=[]),
        #             BackendConfig("tvmcg", enabled=True, features=[]),
        #         ],
        #         features=[
        #             FrameworkFeatureConfig("memplan", framework="utvm", supported=False),
        #         ],
        #     ),
        # ]
        # self.frontends = [
        #     FrontendConfig("saved_model", enabled=False),
        #     FrontendConfig("ipynb", enabled=False),
        #     FrontendConfig(
        #         "tflite",
        #         enabled=True,
        #         features=[
        #             FrontendFeatureConfig("packing", frontend="tflite", supported=False),
        #         ],
        #     ),
        # ]
        # self.vars = {
        #     "TEST": "abc",
        # }
        # self.flags = {}
        # self.platforms = [
        #     PlatformConfig(
        #         "mlif",
        #         enabled=True,
        #         features=[PlatformFeatureConfig("debug", platform="mlif", supported=True)],
        #     )
        # ]
        # self.targets = [
        #     TargetConfig(
        #         "etiss_pulpino",
        #         features=[
        #             TargetFeatureConfig("debug", target="etiss_pulpino", supported=True),
        #             TargetFeatureConfig("attach", target="etiss_pulpino", supported=True),
        #             TargetFeatureConfig("trace", target="etiss_pulpino", supported=True),
        #         ],
        #     ),
        #     TargetConfig(
        #         "host_x86",
        #         features=[
        #             TargetFeatureConfig("debug", target="host_x86", supported=True),
        #             TargetFeatureConfig("attach", target="host_x86", supported=True),
        #         ],
        #     ),
        # ]
