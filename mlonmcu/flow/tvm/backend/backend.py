import os

from ..framework import COMMON_TVM_CONFIG

# from ..support.load_tflite_model import load_tflite_model
from mlonmcu.flow.backend import Backend
from mlonmcu.setup import utils
from .tflite_model_info import get_tflite_model_info


class TVMBackend(Backend):

    registry = {}

    name = None

    FEATURES = ["autotune", "autotuned"]

    DEFAULTS = {
        "opt_level": 3,
        "target_device": None,
        "disabled_passes": [],  # i.e. AlterOpLayout
        "extra_pass_config": {},  # TODO: some example (fuse_max_depth etc.)
    }

    REQUIRED = ["tvm.build_dir", "tvm.pythonpath"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(
            framework="tvm", features=features, config=config, context=context
        )

        self.model = None  # Actual filename!
        self.model_info = None
        self.input_shapes = None

        self.prefix = "model"
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        # TODO: decide if artifacts should be handled by code (str) or file path or binary data

    @property
    def pass_config(self):
        base = {"tir.disable_vectorize": True}
        extra = self.config["extra_pass_config"]
        if isinstance(extra, str):
            import ast

            extra = ast.literal_eval(extra)
        assert isinstance(extra, dict)
        base.update(extra)
        return base

    @property
    def target_device(self):
        return self.config["target_device"]

    @property
    def opt_level(self):
        return self.config["opt_level"]

    def get_pass_config_tvmc_args(self):
        args = []
        for key, value in self.pass_config.items():
            args.extend(["--pass-config", f"{key}={value}"])
        return args

    def get_disabled_pass_tvmc_args(self):
        args = []
        for item in self.config["disabled_passes"]:
            args.extend(["--disable-pass", item])
        return args

    def get_input_shapes_tvmc_args(self):
        if self.input_shapes is None:
            return []
        arg = " ".join(
            [
                f"{name}:[" + ",".join(list(map(str, dims))) + "]"
                for name, dims in self.input_shapes.items()
            ]
        )
        return ["--input-shapes", arg]

    def get_common_tvmc_args(self, executor, fmt="mlf", target="c", runtime="crt"):
        assert executor in ["aot", "graph"], "Unsupported TVM executor"
        return [
            str(self.model),
            "-f",
            fmt,
            "--target",
            target,
            "--executor",
            executor,
            "--runtime",
            runtime,
            *self.get_pass_config_tvmc_args(),
            *self.get_disabled_pass_tvmc_args(),
            *(
                ["--target-c-device", self.target_device]
                if self.target_device is not None
                else []
            ),
            "--opt-level",
            str(self.opt_level),
            *self.get_input_shapes_tvmc_args(),
        ]

    def invoke_tvmc(self, out, command="compile", dump=None):
        args = self.get_tvmc_args()
        args.extend(["--output", str(out)])
        if dump:
            assert isinstance(dump, list)
            args.extend(["--dump-code", ",".join(dump)])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.config["tvm.pythonpath"])
        env["TVM_LIBRARY_PATH"] = str(self.config["tvm.build_dir"])
        verbose = True  # ???
        utils.python("-m", "tvm.driver.tvmc", command, *args, live=verbose, env=env)

    def get_opt_level(self):
        # TODO: Make this a helper function?
        for key, value in self.config:
            if key.split(".")[-1] == "opt_level":
                return int(value)
        return COMMON_TVM_CONFIG["opt_level"]

    def get_target_device(self):
        # TODO: Make this a helper function?
        for key, value in self.config:
            if key.split(".")[-1] == "target_device":
                return int(value)
        return COMMON_TVM_CONFIG["target_device"]

    def get_fuse_max_depth(self):
        # TODO: Make this a helper function?
        for key, value in self.config:
            if key.split(".")[-1] == "fuse_max_depth":
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

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     assert isinstance(cls.name, str)
    #     cls.registry[cls.name] = cls

    def load_model(self, model):
        self.model = model
        with open(model, "rb") as handle:
            model_buf = handle.read()
            self.model_info = get_tflite_model_info(model_buf)
            self.input_shapes = {
                tensor.name: tensor.shape for tensor in self.model_info.inTensors
            }
