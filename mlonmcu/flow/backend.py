from abc import ABC, abstractmethod


class Backend(ABC):

    shortname = None

    def __init__(
        self,
        config={},
        context=None,
    ):
        self.config = config
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
