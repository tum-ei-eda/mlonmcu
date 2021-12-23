from mlonmcu.flow.backend import Backend

class TVMBackend(Backend):

    registry = {}

    shortname = None

    def __init__(self, config={}, context=None):
        super().__init__(config=config, context=context)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.shortname, str)
        cls.registry[cls.shortname] = cls

    def load(self, model):
        self.model = model