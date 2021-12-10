from abc import ABC, abstractmethod

class Backend(ABC):

    def __init__(self, config={}, context=None):
        self.config = config
        self.context = context

    @abstractmethod
    def load(self, model):
        pass

    @abstractmethod
    def generate_code(self):
        pass