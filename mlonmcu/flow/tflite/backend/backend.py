from mlonmcu.flow.backend import Backend


class TFLiteBackend(Backend):

    registry = {}

    name = None

    DEFAULTS = {}

    REQUIRED = []

    def __init__(self, features=None, config=None, context=None):
        super().__init__(framework="tflite", config=config, context=context)
        self.model = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.name, str)
        cls.registry[cls.name] = cls

    def load_model(self, model):
        self.model = model
