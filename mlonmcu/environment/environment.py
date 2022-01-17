from pathlib import Path
import yaml

from .config import *
from .loader import load_environment_from_file
from .writer import write_environment_to_file

from ..feature.feature import FeatureType


def _feature_helper(obj, name):
    features = obj.features
    names = [feature.name for feature in features]
    if name:
        return [features[names.index(name)]]
    else:
        return features


class Environment:
    def __init__(self):
        self._home = None
        self.alias = None
        self.defaults = DefaultsConfig()
        self.paths = {}
        self.repos = {}
        self.frameworks = []
        self.frontends = []
        self.targets = []
        self.vars = {}

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

    def lookup_frontend_feature_configs(self, name=None, frontend=None):
        configs = []
        if frontend:
            names = [frontend.name for frontend in self.frontends]
            index = names.index(frontend)
            assert (
                index is not None
            ), f"Frontend {frontend} not found in environment config"
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
            assert (
                index is not None
            ), f"Framework {framework} not found in environment config"
            configs.extend(_feature_helper(self.frameworks[index], name))
        else:
            for framework in self.frameworks:
                configs.extend(_feature_helper(framework, name))
        return configs

    def lookup_backend_feature_configs(self, name=None, framework=None, backend=None):
        print("backend!", backend)

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
            assert (
                index is not None
            ), f"Framework {framework} not found in environment config"
            if backend:
                names_ = [backend.name for backend in self.frameworks[index].backends]
                index_ = names_.index(backend)
                assert (
                    index_ is not None
                ), f"Backend {backend} not found in environment config"
                configs.extend(
                    _feature_helper(self.frameworks[index].backends[index], name)
                )
            else:
                for backend in self.frameworks[index].backends:
                    configs.extend(_feature_helper(backend, name))
        else:
            for framework in self.frameworks:
                print("backend2", backend)
                if backend:
                    print("backend", backend)
                    names_ = [backend.name for backend in framework.backends]
                    index_ = names_.index(backend)
                    assert (
                        index_ is not None
                    ), f"Backend {backend} not found in environment config"
                    configs.extend(
                        _feature_helper(self.frameworks[index].backends[index], name)
                    )
                else:
                    for backend in framework.backends:
                        configs.extend(_feature_helper(backend, name))
                    backend = None
        return configs

    def lookup_target_feature_configs(self, name=None, target=None):
        configs = []
        if target:
            names = [target.name for target in self.targets]
            index = names.index(backend)
            assert index is not None, f"Target {target} not found in environment config"
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
        target=None,
    ):
        configs = []
        if kind == FeatureType.FRONTEND or kind is None:
            configs.extend(
                self.lookup_frontend_feature_configs(name=name, frontend=frontend)
            )
        if kind == FeatureType.FRAMEWORK or kind is None:
            configs.extend(
                self.lookup_framework_feature_configs(name=name, framework=framework)
            )
        if kind == FeatureType.BACKEND or kind is None:
            configs.extend(
                self.lookup_backend_feature_configs(
                    name=name, framework=framework, backend=backend
                )
            )
        if kind == FeatureType.TARGET or kind is None:
            configs.extend(self.lookup_target_feature_configs(name=name, target=target))

        return configs


class DefaultEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.defaults = DefaultsConfig(
            log_level=LogLevel.DEBUG,
            log_to_file=False,
            default_framework="utvm",
            default_backends={"tflm": "tflmc", "utvm": "tvmaot"},
            default_target="etiss_pulpino",
        )
        self.paths = {
            "deps": PathConfig("./deps"),
            "logs": PathConfig("./logs"),
            "results": PathConfig("./results"),
            "temp": PathConfig("out"),
            "models": [
                PathConfig("./models"),
                PathConfig("/work/models"),
            ],
        }
        self.repos = {
            "tensorflow": RepoConfig(
                "https://github.com/tensorflow/tensorflow.git", ref="v2.5.2"
            ),
            "tflite_micro_compiler": RepoConfig(
                "https://github.com/cpetig/tflite_micro_compiler.git", ref="master"
            ),  # TODO: freeze ref?
            "tvm": RepoConfig(
                "https://github.com/tum-ei-eda/tvm.git", ref="tumeda"
            ),  # TODO: use upstream repo with suitable commit?
            "utvm_staticrt_codegen": RepoConfig(
                "https://github.com/tum-ei-eda/utvm_staticrt_codegen.git", ref="master"
            ),  # TODO: freeze ref?
            "etiss": RepoConfig(
                "https://github.com/tum-ei-eda/etiss.git", ref="master"
            ),  # TODO: freeze ref?
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
                    FrameworkFeatureConfig(
                        "muriscvnn", framework="tflm", supported=False
                    ),
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
                            BackendFeatureConfig(
                                "unpacked_api", backend="tvmaot", supported=True
                            ),
                        ],
                    ),
                    BackendConfig("tvmrt", enabled=True, features=[]),
                    BackendConfig("tvmcg", enabled=True, features=[]),
                ],
                features=[
                    FrameworkFeatureConfig(
                        "memplan", framework="utvm", supported=False
                    ),
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
                    FrontendFeatureConfig(
                        "packing", frontend="tflite", supported=False
                    ),
                ],
            ),
        ]
        self.vars = {
            "TEST": "abc",
        }
        self.targets = [
            TargetConfig(
                "etiss_pulpino",
                features=[
                    TargetFeatureConfig(
                        "debug", target="etiss_pulpino", supported=True
                    ),
                    TargetFeatureConfig(
                        "attach", target="etiss_pulpino", supported=True
                    ),
                    TargetFeatureConfig(
                        "trace", target="etiss_pulpino", supported=True
                    ),
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
        targets=None,
        variables=None,
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
        if targets:
            self.targets = targets
        if variables:
            self.vars = variables
