"""Definitions for TVMFramework."""

from mlonmcu.flow.framework import Framework

# from mlonmcu.flow.tvm import TVMBackend

class TVMFramework(Framework):
    """TVM Framework specialization."""

    name = "tvm"

    FEATURES = []

    DEFAULTS = {}

    REQUIRED = ["tvm.src_dir"]

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)
        # self.backends = TVMBackend.registry

    def get_cmake_args(self):
        args = super().get_cmake_args()
        if "tvm.src_dir" in self.config:
            tvmSrc = self.config["tvm.src_dir"]
        else:
            raise RuntimeError("Can not resolve tvm.src_dir")
        return args + ["-DTVM_SRC=" + str(tvmSrc)]  # TODO: change
