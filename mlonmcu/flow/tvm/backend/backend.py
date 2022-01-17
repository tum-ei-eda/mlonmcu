from ..framework import COMMON_TVM_CONFIG

# from ..support.load_tflite_model import load_tflite_model
from mlonmcu.flow.backend import Backend


class TVMBackend(Backend):

    registry = {}

    name = None

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
        self.process_features()
        self.update_config()

        self.model = None  # Actual filename!

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

    def process_features(self):
        for feature in self.features:
            if FeatureType.BACKEND in feature.types:
                assert (
                    feature.name in self.FEATURES
                ), f"Incompatible backend feature: {feature.name}"
                # TODO: allow incompatible features to mix backends? -> just allow?
                feature.add_backend_config(self.name, self.config)

    def remove_config_prefix(self, config):
        def helper(key):
            return key.split(f"{self.name}.")[-1]

        return {helper(key): value for key, value in config if f"{self.name}." in key}

    def filter_config(self):
        cfg = self.remove_config_prefix(self.config)
        for required in self.REQUIRED:
            value = None
            if required in cfg:
                value = cfg[required]
            elif required in self.config:
                value = self.config[required]
            assert value is not None, f"Required config key can not be None: {required}"

        for key in self.DEFAULTS:
            if key not in cfg:
                cfg[key] = self.DEFAULTS[key]

        for key in cfg:
            if key not in self.DEFAULTS.keys() + self.REQUIRED:
                logger.warn("Backend received an unknown config key: %s", key)
                del cfg[key]

        self.config = cfg

    def get_pass_config_tvmc_args(self):
        pass

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
            "--target-c-device",
            self.device,
            "--opt-level",
            self.opt_level,
            "--input-shapes",
            self.input_shapes,
        ]

    def get_tvmc_args(self):
        # tvmaot
        return self.get_common_tvmc_args("aot") + [
            "--runtime-crt-system-lib",
            str(0),
            "--target-c-constants-byte-alignment",
            str(self.alignment_bytes),
            "--target-c-workspace-byte-alignment",
            str(self.alignment_bytes),
            "--target-c-executor",
            "aot",
            "--target-c-unpacked-api",
            str(int(self.unpacked_api)),
            "--target-c-interface-api",
            "c" if self.unpacked_api else "packed",
        ]

        # tvmrt / tvmgraph?
        # return self.get_common_tvmc_args("graph") + ["--runtime-crt-system-lib", str(1), "--executor-graph-link-params", str(0),
        # tvmcg? -> tvmrt

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
        model_buf = open(path, "rb").read()
        # self.mod, self.params, self.modelInfo = load_tflite_model(model_buf)
