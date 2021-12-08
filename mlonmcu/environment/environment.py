from enum import Enum
from pathlib import Path
import yaml

class LogLevel(Enum):
    DEBUG = 0
    VERBOSE = 1
    INFO = 2
    WARNING = 3
    ERROR = 4

class BaseConfig:

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(vars(self)) + ")"


class DefaultsConfig(BaseConfig):
    # TODO: loglevels enum

    def __init__(self, log_level=LogLevel.INFO, log_to_file=False,
                 default_framework=None, default_backends={}, default_target=None):
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.default_framework = default_framework
        self.default_backends = default_backends
        self.default_target = default_target


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
                self.path = self.base / self.path
        # Resolve symlinks
        self.path = self.path.resolve()


class RepoConfig(BaseConfig):

    def __init__(self, url, ref=None):
        self.url = url
        self.ref = ref

class BackendConfig(BaseConfig):

    def __init__(self, description="", enabled=True, features={}):
        self.description = description
        self.enabled = enabled
        self.features = features


class FeatureKind(Enum):
    UNKNOWN = 0
    FRAMEWORK = 1
    BACKEND = 2
    TARGET = 3
    FRONTEND = 4


class Feature:

    def __init__(self, description="", kind=FeatureKind.UNKNOWN, supported=True):
        self.description = description
        self.type = kind
        self.supported = supported

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(vars(self)) + ")"

class FrameworkFeature(Feature):
    def __init__(self, description="", supported=True):
        super().__init__(description=description, kind=FeatureKind.FRAMEWORK, supported=supported)

class BackendFeature(FrameworkFeature):
    pass

class TargetFeature(Feature):

    def __init__(self, description="", supported=True):
        super().__init__(description=description, kind=FeatureKind.TARGET, supported=supported)

class DebugFeature(TargetFeature):

    def __init__(self, supported=True):
        super().__init__("GDB support", supported=supported)

class TraceFeature(TargetFeature):

    def __init__(self, supported=True):
        super().__init__("Memory trace support", supported=supported)

class FrontendFeature(Feature):

    def __init__(self, description="", supported=True):
        super().__init__(description=description, kind=FeatureKind.FRONTEND, supported=supported)

class FrameworkConfig(BaseConfig):
    def __init__(self, description="", enabled=True, backends={}, features={}):
        self.description = description
        self.enabled = enabled
        self.backends = backends
        self.features = features


class FrontendConfig(BaseConfig):
    def __init__(self, description="", enabled=True, features={}):
        self.description = description
        self.enabled = enabled
        self.features = features

class TargetConfig(BaseConfig):
    def __init__(self, description="", enabled=True, features={}):
        self.description = description
        self.enabled = enabled
        self.features = features




class Environment:

    def __init__(self):
        self._home = None
        self.alias = None
        self.defaults = DefaultsConfig()
        self.paths = {}
        self.repos = {}
        self.frameworks = {}
        self.frontends = {}
        self.targets = {}
        self.vars = {}

    def __str__(self):
        return self.__class__.__name__ + "(" + str(vars(self)) + ")"

    @property
    def home(self):
        """Home directory of mlonmcu environment."""
        return self._home

class DefaultEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self.defaults = DefaultsConfig(
            log_level=LogLevel.DEBUG,
            log_to_file=False,
            default_framework="utvm",
            default_backends={"tflm": "tflmc", "utvm": "tvmaot"},
            default_target="etiss/pulpino",
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
            "tensorflow": RepoConfig("https://github.com/tensorflow/tensorflow.git", ref="v2.5.2"),
            "tflite_micro_compiler": RepoConfig("https://github.com/cpetig/tflite_micro_compiler.git", ref="master"),  # TODO: freeze ref?
            "tvm": RepoConfig("https://github.com/tum-ei-eda/tvm.git", ref="tumeda"),  # TODO: use upstream repo with suitable commit?
            "utvm_staticrt_codegen": RepoConfig("https://github.com/tum-ei-eda/utvm_staticrt_codegen.git", ref="master"),  # TODO: freeze ref?
            "etiss": RepoConfig("https://github.com/tum-ei-eda/etiss.git", ref="master"),  # TODO: freeze ref?
        }
        self.frameworks = {
            "tflm": FrameworkConfig(
                "Tensorflow Lite for Microcontrollers",
                enabled=True,
                backends={
                    "tflmc": BackendConfig("TFLM Compiler", enabled=True, features={}),
                    "tflmi": BackendConfig("TFLM Interpreter", enabled=True, features={}),
                },
                features={
                    "muriscvnn": FrameworkFeature("muRISCV-NN Kernels", supported=False)
                }
            ),
            "utvm": FrameworkConfig(
                "MicroTVM",
                enabled=True,
                backends={
                    "tvmaot": BackendConfig(
                        "TVM Ahead-of-Time Executor",
                        enabled=True,
                        features={
                            "unpacked_api": BackendFeature("Unpacked Interface", supported=True)
                        },
                    ),
                    "tvmrt": BackendConfig(
                        "TVM Graph Runtime",
                        enabled=True,
                        features=[]
                    ),
                    "tvmcg": BackendConfig(
                        "uTVM Staticrt Codegen",
                        enabled=True,
                        features={}
                    ),
                },
                features={
                    "memplan": FrameworkFeature("Custom memory planner", supported=False)
                },
            ),
        }
        self.frontends = {
            "saved_model": FrontendConfig("Tensorflow saved model format (experimental)", enabled=False),
            "ipynb": FrontendConfig("Build TFLite model by running a notebook (experimental)", enabled=False),
            "tflite": FrontendConfig("Use TFLite model flatbuffer",
                enabled=True,
                features={
                    "packing": FrontendFeature("Allows to process flatbuffers with packed/sparse weights", supported=False),
                },
            )

        }
        self.vars = {
            "TEST": "abc",
        }
        self.targets = {
            "etiss/pulpino": TargetConfig(
                "Simple RISC-V Virtual Prototype running in ETISS",
                features={
                    "debug": DebugFeature(True),
                    "trace": TraceFeature(True)
                },
            ),
            "host": TargetConfig(
                "Run target software on local machine (x86)",
                features={
                    "debug": DebugFeature(True),
                },
            ),
        }

class UserEnvironment(DefaultEnvironment):

    def __init__(self, home, merge=False, alias=None, defaults=None, paths=None, repos=None, frameworks=None, frontends=None, targets=None, variables=None):
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
            self.variables = variables

#test_env = DefaultEnvironment()
#print("---")
#print("test_env", test_env)
#print("---")
#print(yaml.dump(vars(test_env)))

#test_file = "/work/git/prj/mlonmcu_open_source/mlonmcu/templates/default.yml.j2"

