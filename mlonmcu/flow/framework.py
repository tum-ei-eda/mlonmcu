from abc import ABC, abstractmethod


class Framework(ABC):
    registry = {}

    shortname = None

    FEATURES = []
    DEFAULTS = {}
    REQUIRED = ["tf.src_dir"]

    def __init__(self, features=None, config=None, backends={}):
        self.features = features if features else []
        self.process_features()
        self.config = config if config else {}
        self.filter_config()
        self.backends = backends  # TODO: get rid of this

    def process_features(self):
        for feature in self.features:
            if FeatureType.FRAMEWORK in feature.types:
                assert (
                    feature.name in self.FEATURES
                ), f"Incompatible framework feature: {feature.name}"
                feature.add_framework_config(self.name, self.config)

    def remove_config_prefix(self, config):
        def helper(key):
            return key.split(f"{self.name}.")[-1]

        return {helper(key): value for key, value in config if f"{self.name}." in key}

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
            if key not in self.DEFAULTS.keys() + self.REQUIRED:
                logger.warn("Backend received an unknown config key: %s", key)
                del cfg[key]

        self.config = cfg

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.shortname, str)
        cls.registry[cls.shortname] = cls

    def get_cmake_args(self):
        assert self.shortname is not None
        return [f"-DFRAMEWORK={self.shortname}"]

    def add_cmake_args(self, args):
        args += self.get_cmake_args()
