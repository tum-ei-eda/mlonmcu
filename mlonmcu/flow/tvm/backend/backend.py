from ..framework import COMMON_TVM_CONFIG
# from ..support.load_tflite_model import load_tflite_model
from mlonmcu.flow.backend import Backend

COMMON_TVM_CONFIG = {
    "opt_level": 3,
    "target_device": None,
    "fuse_max_depth": None,
    "tvm.src_dir": None,  # ???
}


class TVMBackend(Backend):

    registry = {}

    shortname = None

    def __init__(self, features=None, config=None, context=None):
        super().__init__(
            framework="tvm", features=features, config=config, context=context
        )
        self.model = None  # Actual filename!
        self.mod = None
        self.cfg = {"tir.disable_vectorize": True}
        self.opt_level, self.target_device, self.fuse_max_depth = self.get_opt_level(), self.get_target_device(), self.get_fuse_max_depth()  # TODO: unify with other backends
        if self.fuse_max_depth:
            assert self.fuse_max_depth >= 0
            self.cfg["relay.FuseOps.max_depth"] = self.fuse_max_depth

        self.prefix = "model"
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        # TODO: decide if artifacts should be handled by code (str) or file path or binary data

    def get_opt_level(self):
        # TODO: Make this a helper function?
        for key, value in self.config:
            if key.split('.')[-1] == "opt_level":
                return int(value)
        return COMMON_TVM_CONFIG["opt_level"]

    def get_target_device(self):
        # TODO: Make this a helper function?
        for key, value in self.config:
            if key.split('.')[-1] == "target_device":
                return int(value)
        return COMMON_TVM_CONFIG["target_device"]

    def get_fuse_max_depth(self):
        # TODO: Make this a helper function?
        for key, value in self.config:
            if key.split('.')[-1] == "fuse_max_depth":
                return int(value)
        return COMMON_TVM_CONFIG["fuse_max_depth"]


    def get_target_str(self):
        target_str = "c"
        target_str += " --runtime=c"
        if self.target_device:
            target_str += " --device=" + self.target_device
        target_str += f" --model=unknown"  # TODO: required?
        return target_str

    def get_target(self):
        return tvm.target.Target(self.get_target_str())

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.shortname, str)
        cls.registry[cls.shortname] = cls

    def load_model(self, model):
        self.model = model
        model_buf = open(path, "rb").read()
        # self.mod, self.params, self.modelInfo = load_tflite_model(model_buf)

