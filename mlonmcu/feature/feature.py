""" MLonMCU Features API"""

from abc import ABC
from enum import Enum
from typing import List


class FeatureType(Enum):
    OTHER = 0
    SETUP = 1
    FRONTEND = 2
    FRAMEWORK = 3
    BACKEND = 4
    TARGET = 5
    COMPILE = 6


# TODO: features might get an optional context parameter to lookup if they are supported by themselfs in the environment


class FeatureBase(ABC):
    """Feature base class"""

    DEFAULTS = {"enabled": True}
    REQUIRED = []

    def __init__(self, name, config=None):
        print("FeatureBase.__init__")
        self.name = name
        self.config = config if config else {}
        self.filter_config()

    @property
    def enabled(self):
        return bool(self.config.get("enabled", None))

    def remove_config_prefix(self, config):  # TODO: move to different place
        def helper(key):
            return key.split(f"{self.name}.")[-1]

        return {
            helper(key): value
            for key, value in config.items()
            if f"{self.name}." in key
        }

    def filter_config(self):
        cfg = self.remove_config_prefix(self.config)
        for required in self.REQUIRED:
            value = None
            if required in cfg:
                value = cfg[required]
            elif required in self.config:
                value = self.config[required]
            assert value is not None, f"Required config key can not be None: {required}"

        for key in self.DEFAULTS:
            if key not in cfg:
                cfg[key] = self.DEFAULTS[key]

        for key in cfg:
            if key not in list(self.DEFAULTS.keys()) + self.REQUIRED:
                logger.warn("Feature received an unknown config key: %s", key)
                del cfg[key]

        self.config = cfg

    def __repr__(self):
        return type(self).__name__ + f"({self.name})"

    # @property
    # def types(self):
    #     return [base.feature_type for base in type(self).__bases__]

    @classmethod
    def types(cls):
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
        print("FrontendFeature.__init__")

    def get_frontend_config(self, frontend):
        print("get_frontend_config")
        return {}

    def add_frontend_config(self, frontend, config):
        config.update(self.get_frontend_config(frontend))


class FrameworkFeature(FeatureBase):
    """Framework related feature"""

    feature_type = FeatureType.FRAMEWORK

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("FrameworkFeature.__init__")

    def get_framework_config(self, farmework):
        print("get_framework_config")
        return {}

    def add_framework_config(self, framework, config):
        config.update(self.get_framework_config(framework))


class BackendFeature(FeatureBase):
    """Backend related feature"""

    feature_type = FeatureType.BACKEND

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("BackendFeature.__init__")

    def get_backend_config(self, backend):
        print("get_backend_config")
        return {}

    def add_backend_config(self, backend, config):
        # TODO: cfg passed to method instead of contructor or self.config = config
        config.update(self.get_backend_config(backend))


class TargetFeature(FeatureBase):
    """Target related feature"""

    feature_type = FeatureType.TARGET

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("TargetFeature.__init__")

    def get_target_config(self, target):
        print("get_target_config")
        return {}

    def add_target_config(self, target, config):
        # TODO: cfg passed to method instead of contructor or self.config = config
        config.update(self.get_target_config(target))


class CompileFeature(FeatureBase):
    """Compile related feature"""

    feature_type = FeatureType.COMPILE

    def __init__(self, name, config=None):
        super().__init__(name, config=config)
        print("CompileFeature.__init__")

    def get_compile_config(self):
        print("get_compile_config")
        return {}

    def add_compile_config(self, config):
        config.update(self.get_compile_config())

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
        print("SetupFeature.__init__")

    def get_setup_config(self):
        print("get_setup_config")
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


# # registry
#
# REGISTERED_FEATURES = {}
#
# def register_feature(object):
#     def __init__(self, name):
#         self.name
#
#     def __call__(self, cls):
#         assert self.name not in REGISTERED_FEATURES, f"Can not register feature '{self.name}'"
#         REGISTERED_FEATURES[name] = cls
#         return cls
#
#
#
#
# # TODO: get or ADD (update) config?
# # TODO: MLIFFeature? / CompileFeature?
#
#
# def get_available_features(feature_type=None, feature_name=None):
#     print(REGISTERED_FEATURES)
#     if feature_type is None:
#         if feature_name is None:
#             return REGISTERED_FEATURES.values()
#         else:
#             ret = []
#             for name, feature in REGISTERED_FEATURES.items():
#                 if name == feature_name:
#                     ret.append(feature)
#     else:
#         ret = []
#         for feature in REGISTERED_FEATURES.values():
#             if feature_type in feature.types:
#                 if name is None or name == feature_name:
#                     ret.append(feature)
#     return ret


# def lookup_features(name: str) -> List[FeatureBase]:
#     """Get a list of all available features matching a given name.
#
#     Work in progress
#
#     Parameters
#     ----------
#     name : str
#         Name of feature
#
#     Returns
#     -------
#     list
#         List of all features matching the given name
#     """
#     assert name in ALL_FEATURES, f"Unknown feature: {name}"
#     ret = (
#         []
#     )  # For a single feature-name, there can be multiple types of features (e.g. backend vs target) and we want to match all of them
#     if name in FRONTEND_FEATURES:
#         ret.append(FrontendFeature(name))
#     if name in FRAMEWORK_FEATURES:
#         ret.append(FrameworkFeature(name))
#     if name in BACKEND_FEATURES:
#         ret.append(BackendFeature(name))
#     if name in TARGET_FEATURES:
#         ret.append(TargetFeature(name))
#     return ret
