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
        "optimized_kernel_inc_dirs": [],
        "optimized_kernel_libs": [],
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
    def optimized_kernel_libs(self):
        return self.config["optimized_kernel_libs"]

    @property
    def optimized_kernel_inc_dirs(self):
        return self.config["optimized_kernel_inc_dirs"]

    def get_cmake_args(self):
        args = super().get_cmake_args()
        args.append(f"-DTF_SRC={self.tf_src}")
        if self.optimized_kernel:
            args.append(f"-DTFLM_OPTIMIZED_KERNEL={self.optimized_kernel}")
        if self.optimized_kernel_inc_dirs:
            temp = "\;".join(self.optimized_kernel_inc_dirs)
            args.append(f"-DTFLM_OPTIMIZED_KERNEL_INCLUDE_DIR={temp}")
        if self.optimized_kernel_libs:
            temp = "\;".join(self.optimized_kernel_libs)
            args.append(f"-DTFLM_OPTIMIZED_KERNEL_LIB={temp}")
        return args

    # TODO: get_cmake_args -> get_plaform_vars (dict instead of list of strings)
    def get_espidf_defs(self):
        if self.extra_incs or self.extra_libs:
            raise NotImplementedError("Extra incs or libs are currently not supported for esp-idf")
        return {"TF_DIR": str(self.tf_src)}
