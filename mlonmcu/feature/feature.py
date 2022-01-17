""" MLonMCU Features API"""

from abc import ABC
from enum import Enum
from typing import List


class FeatureType(Enum):
    OTHER = 0
    FRONTEND = 1
    FRAMEWORK = 2
    BACKEND = 3
    TARGET = 4
    COMPILE = 5


# TODO: features might get an optional context parameter to lookup if they are supported by themselfs in the environment


class FeatureBase(ABC):
    """Feature base class"""

    def __init__(self, name, config=None):
        print("FeatureBase.__init__")
        self.name = name
        self.config = config if config else {}

    def __repr__(self):
        return type(self).__name__ + f"({self.name})"

    @property
    def types(self):
        return [base.feature_type for base in type(self).__bases__]

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


class FrontendFeature(FeatureBase):
    """Frontend related feature"""

    feature_type = FeatureType.FRONTEND

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("FrontendFeature.__init__")

    def get_frontend_config(self, frontend):
        print("get_frontend_config")
        return {}


class FrameworkFeature(FeatureBase):
    """Framework related feature"""

    feature_type = FeatureType.FRAMEWORK

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("FrameworkFeature.__init__")

    def get_framework_config(self, farmework):
        print("get_framework_config")
        return {}


class BackendFeature(FeatureBase):
    """Backend related feature"""

    feature_type = FeatureType.BACKEND

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("BackendFeature.__init__")

    def get_backend_config(self, backend):
        print("get_backend_config")
        return {}

    # TODO: this IDEA
    def add_backend_config(self, backend, config):
        # TODO: cfg passed to method instead of contructor or self.config = config
        return config.update(self.get_backend_config(backend))


class TargetFeature(FeatureBase):
    """Target related feature"""

    feature_type = FeatureType.TARGET

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("TargetFeature.__init__")

    def get_target_config(self, target):
        print("get_target_config")
        return {}


class CompileFeature(FeatureBase):
    """Compile related feature"""

    feature_type = FeatureType.COMPILE

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("CompileFeature.__init__")

    def get_compile_config(self):
        print("get_compile_config")
        return {}


# TODO: get or ADD (update) config?
# TODO: MLIFFeature? / CompileFeature?


def lookup_features(name: str) -> List[FeatureBase]:
    """Get a list of all available features matching a given name.

    Work in progress

    Parameters
    ----------
    name : str
        Name of feature

    Returns
    -------
    list
        List of all features matching the given name
    """
    assert name in ALL_FEATURES, f"Unknown feature: {name}"
    ret = (
        []
    )  # For a single feature-name, there can be multiple types of features (e.g. backend vs target) and we want to match all of them
    if name in FRONTEND_FEATURES:
        ret.append(FrontendFeature(name))
    if name in FRAMEWORK_FEATURES:
        ret.append(FrameworkFeature(name))
    if name in BACKEND_FEATURES:
        ret.append(BackendFeature(name))
    if name in TARGET_FEATURES:
        ret.append(TargetFeature(name))
    return ret
