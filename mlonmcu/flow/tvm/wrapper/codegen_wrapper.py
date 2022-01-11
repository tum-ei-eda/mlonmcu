from abc import ABC

# Abstract class
class CodegenWrapper(ABC):
    def __init__(self, modelInfo, TODO):
        pass

    @abstractmethod
    def gen_wrapper(self):
        raise NotImplementedError
        # TODO

    def generate_wrapper(self, filename):
        with open(filename) as f:
            f.write(self.gen_wrapper())
