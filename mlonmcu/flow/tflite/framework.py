from mlonmcu.flow.framework import Framework
from mlonmcu.flow.tflite import TFLiteBackend


class TFLiteFramework(Framework):

    shortname = "tflm"
    backends = TFLiteBackend.registry

    def __init__(self):
        super().__init__()
