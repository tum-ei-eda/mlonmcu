
from enum import Enum

class ModelFormat(Enum):

    UNKNOWN = 0
    TFLITE = 1

class Model:

    def __init__(self, name, path, alt=None, fmt=ModelFormat.TFLITE, metadata=None):
        self.name = name
        self.path = path
        self.alt = alt
        self.format = fmt
        self.metadata = metadata

    def __repr__(self):
        if self.alt:
            return f"Model({self.name},alt={self.alt},path={self.path})"
        return f"Model({self.name},path={self.path})"