"""Definitions for TFLiteFramework."""

from pathlib import Path

from mlonmcu.flow.framework import Framework
from mlonmcu.flow.tflite import TFLiteBackend


class TFLiteFramework(Framework):
    """TFLite Framework specialization."""

    name = "tflite"

    FEATURES = ["muriscvnn", "cmsisnn"]

    DEFAULTS = {
        "optimized_kernel": None,
        "optimized_kernel_inc_dir": None,
        "optimized_kernel_lib": None,
    }

    REQUIRED = ["tf.src_dir"]

    backends = TFLiteBackend.registry

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)

    @property
    def tf_src(self):
        return Path(self.config["tf.src_dir"])

    @property
    def optimized_kernel(self):
        return self.config["optimized_kernel"]

    @property
    def optimized_kernel_lib(self):
        return self.config["optimized_kernel_lib"]

    @property
    def optimized_kernel_inc_dir(self):
        return self.config["optimized_kernel_inc_dir"]

    def get_cmake_args(self):
        args = super().get_cmake_args()
        args.append(f"-DTF_SRC={self.tf_src}")
        if self.optimized_kernel:
            args.append(f"-DTFLM_OPTIMIZED_KERNEL={self.optimized_kernel}")
        if self.optimized_kernel_inc_dir:
            args.append(f"-DTFLM_OPTIMIZED_KERNEL_INCLUDE_DIR={self.optimized_kernel_inc_dir}")
        if self.optimized_kernel_lib:
            args.append(f"-DTFLM_OPTIMIZED_KERNEL_LIB={self.optimized_kernel_lib}")
        return args
