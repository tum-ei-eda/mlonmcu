from abc import ABC, abstractmethod


class Backend(ABC):

    shortname = None

    def __init__(
        self,
        framework="",
        features=None,
        config=None,
        context=None,
    ):
        self.framework = framework
        self.features = features if features else []
        self.config = config if config else {}
        self.context = context

    def __repr__(self):
        name = type(self).shortname
        return f"Backend({name})"

    @abstractmethod
    def load(self, model):
        pass

    @abstractmethod
    def generate_code(self):
        pass

    def get_cmake_args(self):
        return []
