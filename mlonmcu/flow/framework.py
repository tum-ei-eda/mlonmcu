from abc import ABC, abstractmethod

from mlonmcu.feature.type import FeatureType
from mlonmcu.config import filter_config
from mlonmcu.feature.features import get_matching_features


class Framework(ABC):
    registry = {}

    name = None

    FEATURES = []
    DEFAULTS = {}
    REQUIRED = ["tf.src_dir"]

    def __init__(self, features=None, config=None, backends={}):
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.REQUIRED)
        self.backends = backends  # TODO: get rid of this

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.FRAMEWORK)
        for feature in features:
            assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
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

    def get_cmake_args(self):
        assert self.name is not None
        return [f"-DFRAMEWORK={self.name}"]

    def add_cmake_args(self, args):
        args += self.get_cmake_args()

    def get_espidf_defs(self):
        return {"MLONMCU_FRAMEWORK": self.name}

    def add_espidf_defs(self, defs):
        defs.update(self.get_espidf_defs())
