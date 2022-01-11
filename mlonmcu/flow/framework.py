from abc import ABC


class Framework(ABC):
    registry = {}

    shortname = None

    def __init__(self, backends={}):
        self.backends = backends

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.shortname, str)
        cls.registry[cls.shortname] = cls
