from mlonmcu.flow.framework import Framework

# from mlonmcu.flow.tvm import TVMBackend

COMMON_TVM_CONFIG = {
    "opt_level": 3,
    "target_device": None,
    "mod_prefix": "model",
}


class TVMFramework(Framework):

    shortname = "tvm"

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
