from abc import ABC, abstractmethod


class Framework(ABC):
    registry = {}

    shortname = None

    def __init__(self, features=None, config=None, backends={}):
        self.features = features if features else []
        self.config = config if config else {}
        self.backends = backends  # TODO: get rid of this

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.shortname, str)
        cls.registry[cls.shortname] = cls

    def get_cmake_args(self):
        assert self.shortname is not None
        return [f"-DFRAMEWORK={self.shortname}"]
