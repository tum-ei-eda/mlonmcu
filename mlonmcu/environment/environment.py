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

from .config import (
    DefaultsConfig,
    PathConfig,
    RepoConfig,
    FrameworkConfig,
    FrameworkFeatureConfig,
    BackendConfig,
    BackendFeatureConfig,
    TargetConfig,
    TargetFeatureConfig,
    PlatformConfig,
    PlatformFeatureConfig,
    FrontendConfig,
    FrontendFeatureConfig,
)
from .loader import load_environment_from_file
from .writer import write_environment_to_file

from mlonmcu.feature.type import FeatureType


def _feature_helper(obj, name):
    if not obj.enabled:
        return []
    features = obj.features
    if name:
        return [feature for feature in features if feature.name == name]
    else:
        return features


def _extract_names(objs):
    return [obj.name for obj in objs]


def _filter_enabled(objs):
    return [obj for obj in objs if obj.enabled]


class Environment:
    def __init__(self):
        self._home = None
        self.alias = None
        self.defaults = DefaultsConfig()
        self.paths = {}
        self.repos = {}
        self.frameworks = []
        self.frontends = []
        self.platforms = []
        self.toolchains = []
        self.targets = []
        self.vars = {}
        self.flags = {}

    def __str__(self):
        return self.__class__.__name__ + "(" + str(vars(self)) + ")"

    @property
    def home(self):
        """Home directory of mlonmcu environment."""
        return self._home

    @classmethod
    def from_file(cls, filename):
        return load_environment_from_file(filename, base=cls)

    def to_file(self, filename):
        write_environment_to_file(self, filename)

    def lookup_path(self, name):
        assert name in self.paths, f"Unable to find '{name}' path in environment config"
        return self.paths[name]

    def lookup_var(self, name, default=None):
        return self.vars.get(name, default)

    def lookup_frontend_feature_configs(self, name=None, frontend=None):
        configs = []
        if frontend:
            names = [frontend.name for frontend in self.frontends]
            index = names.index(frontend)
            assert index is not None, f"Frontend {frontend} not found in environment config"
            configs.extend(_feature_helper(self.frontends[index], name))
        else:
            for frontend in self.frontends:
                configs.extend(_feature_helper(frontend, name))
        return configs

    def lookup_framework_feature_configs(self, name=None, framework=None):
        configs = []
        if framework:
            names = [framework.name for framework in self.frameworks]
            index = names.index(framework)
            assert index is not None, f"Framework {framework} not found in environment config"
            configs.extend(_feature_helper(self.frameworks[index], name))
        else:
            for framework in self.frameworks:
                configs.extend(_feature_helper(framework, name))
        return configs

    def lookup_backend_feature_configs(self, name=None, framework=None, backend=None):
        def helper(framework, backend, name):
            backend_features = self.framework[framework].backends[backend].features
            if name:
                return [backend_features[name]]
            else:
                return backend_features.values()

        configs = []
        if framework:
            names = [framework.name for framework in self.frameworks]
            index = names.index(framework)
            assert index is not None, f"Framework {framework} not found in environment config"
            if backend:
                names_ = [backend.name for backend in self.frameworks[index].backends]
                index_ = names_.index(backend)
                assert index_ is not None, f"Backend {backend} not found in environment config"
                configs.extend(_feature_helper(self.frameworks[index].backends[index], name))
            else:
                for backend in self.frameworks[index].backends:
                    configs.extend(_feature_helper(backend, name))
        else:
            for framework in self.frameworks:
                if backend:
                    names_ = [backend.name for backend in framework.backends]
                    index_ = names_.index(backend)
                    assert index_ is not None, f"Backend {backend} not found in environment config"
                    configs.extend(_feature_helper(self.frameworks[index].backends[index], name))
                else:
                    for backend in framework.backends:
                        configs.extend(_feature_helper(backend, name))
                    backend = None
        return configs

    def lookup_platform_feature_configs(self, name=None, platform=None):
        configs = []
        if platform:
            names = [platform.name for platform in self.platforms]
            index = names.index(platform)
            assert (
                index is not None
            ), f"Platform {platform} not found in environment config"  # TODO: do not fail, just return empty list
            configs.extend(_feature_helper(self.platforms[index], name))
        else:
            for platform in self.platforms:
                configs.extend(_feature_helper(platform, name))
        return configs

    def lookup_target_feature_configs(self, name=None, target=None):
        configs = []
        if target:
            names = [target.name for target in self.targets]
            index = names.index(target)
            assert (
                index is not None
            ), f"Target {target} not found in environment config"  # TODO: do not fail, just return empty list
            configs.extend(_feature_helper(self.targets[index], name))
        else:
            for target in self.targets:
                configs.extend(_feature_helper(target, name))
        return configs

    def lookup_feature_configs(
        self,
        name=None,
        kind=None,
        frontend=None,
        framework=None,
        backend=None,
        platform=None,
        target=None,
    ):
        configs = []
        if kind == FeatureType.FRONTEND or kind is None:
            configs.extend(self.lookup_frontend_feature_configs(name=name, frontend=frontend))
        if kind == FeatureType.FRAMEWORK or kind is None:
            configs.extend(self.lookup_framework_feature_configs(name=name, framework=framework))
        if kind == FeatureType.BACKEND or kind is None:
            configs.extend(self.lookup_backend_feature_configs(name=name, framework=framework, backend=backend))
        if kind == FeatureType.PLATFORM or kind is None:
            configs.extend(self.lookup_platform_feature_configs(name=name, platform=platform))
        if kind == FeatureType.TARGET or kind is None:
            configs.extend(self.lookup_target_feature_configs(name=name, target=target))

        return configs

    def supports_feature(self, name):
        configs = self.lookup_feature_configs(name=name)
        supported = [feature.supported for feature in configs]
        return any(supported)

    def has_feature(self, name):
        """An alias for supports_feature."""
        return self.supports_feature(name)

    def lookup_backend_configs(self, backend=None, framework=None, names_only=False):
        enabled_frameworks = _filter_enabled(self.frameworks)

        configs = []
        for framework_config in enabled_frameworks:
            if framework is not None and framework_config.name != framework:
                continue
            enabled_backends = _filter_enabled(framework_config.backends)
            if backend is None:
                configs.extend(enabled_backends)
            else:
                for backend_config in enabled_backends:
                    if backend_config.name == backend:
                        return [backend_config.name if names_only else backend_config]
        return _extract_names(configs) if names_only else configs

    def lookup_framework_configs(self, framework=None, names_only=False):
        enabled_frameworks = _filter_enabled(self.frameworks)

        if framework is None:
            return _extract_names(enabled_frameworks) if names_only else enabled_frameworks

        for framework_config in enabled_frameworks:
            if framework_config.name == framework:
                return [framework_config.name if names_only else framework_config]

        return []

    def lookup_frontend_configs(self, frontend=None, names_only=False):
        enabled_frontends = _filter_enabled(self.frontends)

        if frontend is None:
            return _extract_names(enabled_frontends) if names_only else enabled_frontends

        for frontend_config in enabled_frontends:
            if frontend_config.name == frontend:
                return [frontend_config.name if names_only else frontend_config]
        return []

    def lookup_platform_configs(self, platform=None, names_only=False):
        enabled_platforms = _filter_enabled(self.platforms)

        if platform is None:
            return _extract_names(enabled_platforms) if names_only else enabled_platforms
        for platform_config in enabled_platforms:
            if platform_config.name == platform:
                return [platform_config.name if names_only else platform_config]

        return []

    def lookup_target_configs(self, target=None, names_only=False):
        enabled_targets = _filter_enabled(self.targets)

        if target is None:
            return _extract_names(enabled_targets) if names_only else enabled_targets

        for target_config in enabled_targets:
            if target_config.name == target:
                return [target_config.name if names_only else target_config]
        return []

    def has_frontend(self, name):
        configs = self.lookup_frontend_configs(frontend=name)
        return len(configs) > 0

    def has_backend(self, name):
        configs = self.lookup_backend_configs(backend=name)
        return len(configs) > 0

    def has_framework(self, name):
        configs = self.lookup_framework_configs(framework=name)
        return len(configs) > 0

    def has_platform(self, name):
        configs = self.lookup_platform_configs(platform=name)
        return len(configs) > 0

    def has_toolchain(self, name):
        return self.toolchains.get(name, False)

    def has_target(self, name):
        configs = self.lookup_target_configs(target=name)
        return len(configs) > 0

    # TODO: actually we do not need to explicitly enable those? environment.yml list the default enabled ones instead
    # of the supported ones in the environment
    # def has_postprocess(self, name):
    #     configs = self.lookup_postprocess_configs(postprocess=name)
    #     return len(configs) > 0

    def get_default_backends(self, framework):
        if framework is None or framework not in self.defaults.default_backends:
            return []
        default = self.defaults.default_backends[framework]
        # framework_names = [framework_config.name for framework_config in self.frameworks]
        # framework_config = self.frameworks[framework_names.index(framework)]
        if default is None:
            return []
        if isinstance(default, str):
            if default == "*":  # Wildcard all enabled frameworks
                default = self.get_enabled_backends()
            else:
                default = [default]
        else:
            assert isinstance(default, list), "TODO"
        return default

    def get_default_frameworks(self):
        default = self.defaults.default_framework
        if default is None:
            return []
        if isinstance(default, str):
            if default == "*":  # Wildcard all enabled frameworks
                default = self.get_enabled_frameworks()
            else:
                default = [default]
        else:
            assert isinstance(default, list), "TODO"
        return default

    def get_default_targets(self):
        default = self.defaults.default_target
        if default is not None:
            if isinstance(default, str):
                if default == "*":  # Wildcard all enabled targets
                    default = self.get_enabled_targets()
                else:
                    default = [default]
            else:
                assert isinstance(default, list)
        return default


class DefaultEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.defaults = DefaultsConfig(
            log_level=logging.DEBUG,
            log_to_file=False,
            default_framework=None,
            default_backends={},
            default_target=None,
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
        self.frameworks = [
            FrameworkConfig(
                "tflm",
                enabled=True,
                backends=[
                    BackendConfig("tflmc", enabled=True, features=[]),
                    BackendConfig("tflmi", enabled=True, features=[]),
                ],
                features=[
                    FrameworkFeatureConfig("muriscvnn", framework="tflm", supported=False),
                ],
            ),
            FrameworkConfig(
                "utvm",
                enabled=True,
                backends=[
                    BackendConfig(
                        "tvmaot",
                        enabled=True,
                        features=[
                            BackendFeatureConfig("unpacked_api", backend="tvmaot", supported=True),
                        ],
                    ),
                    BackendConfig("tvmrt", enabled=True, features=[]),
                    BackendConfig("tvmcg", enabled=True, features=[]),
                ],
                features=[
                    FrameworkFeatureConfig("memplan", framework="utvm", supported=False),
                ],
            ),
        ]
        self.frontends = [
            FrontendConfig("saved_model", enabled=False),
            FrontendConfig("ipynb", enabled=False),
            FrontendConfig(
                "tflite",
                enabled=True,
                features=[
                    FrontendFeatureConfig("packing", frontend="tflite", supported=False),
                ],
            ),
        ]
        self.vars = {
            "TEST": "abc",
        }
        self.flags = {}
        self.platforms = [
            PlatformConfig(
                "mlif",
                enabled=True,
                features=[PlatformFeatureConfig("debug", platform="mlif", supported=True)],
            )
        ]
        self.toolchains = {}
        self.targets = [
            TargetConfig(
                "etiss_pulpino",
                features=[
                    TargetFeatureConfig("debug", target="etiss_pulpino", supported=True),
                    TargetFeatureConfig("attach", target="etiss_pulpino", supported=True),
                    TargetFeatureConfig("trace", target="etiss_pulpino", supported=True),
                ],
            ),
            TargetConfig(
                "host_x86",
                features=[
                    TargetFeatureConfig("debug", target="host_x86", supported=True),
                    TargetFeatureConfig("attach", target="host_x86", supported=True),
                ],
            ),
        ]


class UserEnvironment(DefaultEnvironment):
    def __init__(
        self,
        home,
        merge=False,
        alias=None,
        defaults=None,
        paths=None,
        repos=None,
        frameworks=None,
        frontends=None,
        platforms=None,
        toolchains=None,
        targets=None,
        variables=None,
        default_flags=None,
    ):
        super().__init__()
        self._home = home

        if merge:
            raise NotImplementedError

        if alias:
            self.alias = alias
        if defaults:
            self.defaults = defaults
        if paths:
            self.paths = paths
        if repos:
            self.repos = repos
        if frameworks:
            self.frameworks = frameworks
        if frontends:
            self.frontends = frontends
        if platforms:
            self.platforms = platforms
        if toolchains:
            self.toolchains = toolchains
        if targets:
            self.targets = targets
        if variables:
            self.vars = variables
        if default_flags:
            self.flags = default_flags
