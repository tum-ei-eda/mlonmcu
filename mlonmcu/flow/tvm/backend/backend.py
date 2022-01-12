from mlonmcu.flow.backend import Backend


class TVMBackend(Backend):

    registry = {}

    shortname = None

    def __init__(self, features=None, config=None, context=None):
        super().__init__(
            framework="tvm", features=features, config=config, context=context
        )
        self.model = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.shortname, str)
        cls.registry[cls.shortname] = cls

    def load(self, model):
        self.model = model

    def get_cmake_args(self):
        assert self.shortname is not None
        return [f"-DBACKEND={self.shortname}"]
