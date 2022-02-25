"""Definitions for TVMFramework."""

from pathlib import Path

from mlonmcu.flow.framework import Framework

# from mlonmcu.flow.tvm import TVMBackend


class TVMFramework(Framework):
    """TVM Framework specialization."""

    name = "tvm"

    FEATURES = ["cmsisnnbyoc"]

    DEFAULTS = {
        "extra_incs": [],
        "extra_libs": [],
    }

    REQUIRED = ["tvm.src_dir"]

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)
        # self.backends = TVMBackend.registry

    @property
    def tvm_src(self):
        return Path(self.config["tvm.src_dir"])

    @property
    def extra_incs(self):
        return self.config["extra_incs"]

    @property
    def extra_libs(self):
        return self.config["extra_libs"]

    def get_cmake_args(self):
        args = super().get_cmake_args()
        if self.extra_incs:
            temp = "\;".join(self.extra_incs)
            args.append(f"-DTVM_EXTRA_INCS={temp}")
        if self.extra_libs:
            temp = "\;".join(self.extra_libs)
            args.append(f"-DTVM_EXTRA_LIBS={temp}")
        return args + ["-DTVM_SRC=" + str(self.tvm_src)]  # TODO: change

    # TODO: get_cmake_args -> get_plaform_vars (dict instead of list of strings)
    def get_espidf_defs(self):
        if self.extra_incs or self.extra_libs:
            raise NotImplementedError("Extra incs or libs are currently not supported for esp-idf")
        return {"TVM_DIR": str(self.tvm_src)}
