

class DebugArena(BackendFeature, CompileFeature):
    def __init__(self, config=None):
        super().__init__("debug_arena", config=config)

    def get_backend_config(self, backend):
        assert backend in ["tvmaot", "tvmcg", "tvmrt"], TODO
        return {f"{backend}.debug_arena": True}

    def get_compile_config(self):
        return {"mlif.debug_arena": True}

class Muriscvnn(SetupFeature, FrameworkFeature, CompileFeature):
    ???

    def get_backend_config(self, backend):
        assert backend in ["tflmc", "tflmi"], TODO
        # TODO: how to handle multiple kernels? (list -> comma separated string?)
        return {f"{backend}.extra_kernel": "muriscvnn"}

    def get_compile_config(self):
        return {"mlif."}

    def get_setup_flags(self):
        return {"tflmc.exe": ["muriscvnn"]}

    if tflmc
        "tflmc.exe": cache["tflmc.exe", ("muriscvnn")]

class Debug(SetupFeature, CompileFeature):

    def get_required_cache_flags(self):
        return {"*": ["debug"]} # what about caches which do not have a debug version? Hardcoding would b tricky as well due to inter feature_dependencies ()
    TODO: or dbg (defined in tasks.py)???
debug.enabled

class UnpackedApi # TODO: should this be a feature or config only?
unpacked_api.enabled?

class Packed(FrameworkFeature, FrontendFeature, BackendFeature, ?)
-> CompileFeature because of cmake flags/defines?
packed.enabled?

class Packing(FrontendFeature)
only packing, not using?
packing.enabled

class Fallback(FrameworkFeature, BackendFeature(?), CompileFeature)
-> hard to model..., preprocess for tflmc?
fallback.enable
fallback.config_file
VS
fallback.ops
fallback.mode
fallback.names_file
fallback.merge_compiler_regions

class Memplan(FrameworkFeature)
-> enable this via backend
memplan.enabled

class Visualize()
visualize.enabled
visualize.mode

-> Setup feature to combine flags?

class Autotuned(
    BackendFeature
):  # FronendFeature to collect tuning logs or will we store them somewhere else?
    def __init__(self, config=None):
        super().__init__("autotuned", config=config)

    def get_backend_config(self, backend):
        assert backend in ["tvmaot", "tvmcg", "tvmrt"]  # TODO: backend in TVM_BACKENDS
        # TODO: error handling her eor on backend?
        return {f"{backend}.autotuning_results": self.config.get("autotuned.results", None)}


class Autotune(
    BackendFeature
):  # TODO: how to model that Autotune might depend on Autotuned or the otehr way around?
    def __init__(self, config=None):
        super().__init__("autotune", config=config)

    def get_backend_config(self, backend):
        assert backend in ["tvmaot", "tvmcg", "tvmrt"]  # TODO: backend in TVM_BACKENDS
        if "tvm.autotuning_results" in self.config:
            results = Path(self.config["tvm.autotuning_results"])
        else:
            raise RuntimeError(
                "Missing config value 'tvm.autotuning_results' for feature 'autotuned'"
            )
            # TODO: figure out a default path automatically
        return {
            f"{backend}.autotuning_enable": True)
            f"{backend}.autotuning_results": self.config.get("autotune.results", None)
            f"{backend}.autotuning_append": self.config.get("autotune.append", None),
            f"{backend}.autotuning_tuner": self.config.get("autotune.tuner", None),
            f"{backend}.autotuning_trials": self.config.get("autotune.trials", None),
            f"{backend}.autotuning_early_stopping": self.config.get("autotune.early_stopping", None),
            f"{backend}.autotuning_num_workers": self.config.get("autotune.num_workers", None),
            f"{backend}.autotuning_max_parallel": self.config.get("autotune.max_parallel", None),
        }

# Frontend features
TFLITE_FRONTEND_FEATURES = ["packing"]
FRONTEND_FEATURES = TFLITE_FRONTEND_FEATURES

# Framework features
TFLITE_FRAMEWORK_FEATURES = ["packing", "muriscvnn"]
TVM_FRAMEWORK_FEATURES = ["autotuning"]
FRAMEWORK_FEATURES = TFLITE_FRAMEWORK_FEATURES + TVM_FRAMEWORK_FEATURES

# Backend features
TFLITE_BACKEND_FEATURES = []
TVMAOT_BACKEND_FEATURES = ["unpacked_api"]
TVM_BACKEND_FEATURES = TVMAOT_BACKEND_FEATURES
BACKEND_FEATURES = TFLITE_BACKEND_FEATURES + TVM_BACKEND_FEATURES

# Traget features
TARGET_FEATURES = ["trace"]

ALL_FEATURES = (
    FRONTEND_FEATURES + FRAMEWORK_FEATURES + BACKEND_FEATURES + TARGET_FEATURES
)
