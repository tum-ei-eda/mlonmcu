from enum import Enum
from collections import namedtuple

ModelFormat = namedtuple("ModelFormat", ["value", "extensions"])


class ModelFormats(Enum):
    @property
    def extensions(self):
        return self.value.extensions

    @property
    def extension(self):
        return self.value.extensions[0]

    NONE = ModelFormat(0, [])
    TFLITE = ModelFormat(1, ["tflite"])
    PACKED = ModelFormat(2, ["tflm"])
    IPYNB = ModelFormat(3, ["ipynb"])
    ONNX = ModelFormat(4, ["onnx"])


class Model:
    def __init__(self, name, paths, alt=None, formats=ModelFormats.TFLITE, metadata=None):
        self.name = name
        self.paths = paths
        if not isinstance(self.paths, list):
            self.paths = [self.path]
        self.alt = alt
        self.formats = formats
        if not isinstance(self.formats, list):
            self.formats = [formats]
        self.metadata = metadata

    def __repr__(self):
        if self.alt:
            return f"Model({self.name},alt={self.alt})"
        return f"Model({self.name})"
