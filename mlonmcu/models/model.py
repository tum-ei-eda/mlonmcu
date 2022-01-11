from enum import Enum


class ModelFormat(Enum):
    NONE = 0
    TFLITE = 1
    PACKED = 2
    IPYNB = 3
    ONNX = 4


class Model:
    def __init__(self, name, path, alt=None, format=ModelFormat.TFLITE, metadata=None):
        self.name = name
        self.path = path
        self.alt = alt
        self.format = format
        self.metadata = metadata

    def __repr__(self):
        if self.alt:
            return f"Model({self.name},alt={self.alt})"
        return f"Model({self.name})"
