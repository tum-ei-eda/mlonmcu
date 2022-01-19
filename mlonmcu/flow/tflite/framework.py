"""Definitions for TFLiteFramework."""

from mlonmcu.flow.framework import Framework
from mlonmcu.flow.tflite import TFLiteBackend


class TFLiteFramework(Framework):
    """TFLite Framework specialization."""

    name = "tflite"
    backends = TFLiteBackend.registry

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)

    def get_cmake_args(self):
        args = super().get_cmake_args()
        if "tf.src_dir" in self.config:
            tfSrc = self.config["tf.src_dir"]
        else:
            raise RuntimeError("Can not resolve tf.src_dir")
        return args + ["-DTF_SRC=" + str(tfSrc)]  # TODO: change
