#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Definition of MLonMCU features and the feature registry."""

from pathlib import Path
from .feature import (
    BackendFeature,
    FrameworkFeature,
    PlatformFeature,
    FrontendFeature,
    TargetFeature,
    SetupFeature,
    RunFeature,
    FeatureBase,
)

from mlonmcu.utils import is_power_of_two


def filter_none(data):
    """Helper function which drop dict items with a None value."""
    assert isinstance(data, dict), "Dict only"
    out = {key: value for key, value in data.items() if value is not None}
    return out


REGISTERED_FEATURES = {}


def register_feature(name):
    """Decorator for adding a feature to the global registry."""

    def real_decorator(obj):
        REGISTERED_FEATURES[name] = obj

    return real_decorator


def get_available_feature_names(feature_type=None):
    """Utility for getting feature names."""
    ret = []
    if feature_type is None:
        return REGISTERED_FEATURES.keys()

    for name, feature in REGISTERED_FEATURES.items():
        if feature_type in list(feature.types()):
            ret.append(name)
    return ret


def get_available_features(feature_type=None, feature_name=None):
    """Utility for looking up features."""
    names = get_available_feature_names(feature_type=feature_type)
    return [REGISTERED_FEATURES[name] for name in names if feature_name is None or name == feature_name]


def get_matching_features(features, feature_type):
    return [feature for feature in features if feature_type in feature.types()]


@register_feature("debug_arena")
class DebugArena(BackendFeature, PlatformFeature):
    """Enable verbose printing of arena usage for debugging."""

    def __init__(self, config=None):
        super().__init__("debug_arena", config=config)

    def get_backend_config(self, backend):
        assert backend in [
            "tvmaot",
            "tvmcg",
            "tvmrt",
        ], f"Unsupported feature '{self.name}' for backend '{backend}'"
        # TODO: TFLM also compatible?
        return {f"{backend}.debug_arena": self.enabled}

    def get_platform_defs(self, platform):
        if platform == "espidf":
            val = "y" if self.enabled else "n"  # TODO: bool to string at later step?
        else:
            val = "ON" if self.enabled else "OFF"
        return {"DEBUG_ARENA": val}


@register_feature("validate")
class Validate(FrontendFeature, PlatformFeature):
    """Enable validaton of inout and output tensors."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "allow_missing": True,
        "fail_on_error": None,
    }

    def __init__(self, config=None):
        super().__init__("validate", config=config)

    @property
    def allow_missing(self):
        return bool(self.config["allow_missing"])

    @property
    def fail_on_error(self):
        return self.config["fail_on_error"]

    def get_frontend_config(self, frontend):
        if not self.allow_missing:
            raise NotImplementedError
        return {f"{frontend}.use_inout_data": True}

    def get_platform_config(self, platform):
        assert platform == "mlif", f"Unsupported feature '{self.name}' for platform '{platform}'"
        return filter_none({
            f"{platform}.ignore_data": False,
            f"{platform}.fail_on_error": self.fail_on_error,
        })


@register_feature("muriscvnn")
class Muriscvnn(SetupFeature, FrameworkFeature):
    """MuriscvNN CMSIS-NN wrappers for TFLite Micro"""

    REQUIRED = ["muriscvnn.lib", "muriscvnn.inc_dir"]

    def __init__(self, config=None):
        super().__init__("muriscvnn", config=config)

    @property
    def muriscvnn_lib(self):
        return str(self.config["muriscvnn.lib"])

    @property
    def muriscvnn_inc_dir(self):
        return str(self.config["muriscvnn.inc_dir"])

    def add_framework_config(self, framework, config):
        assert framework == "tflite", f"Unsupported feature '{self.name}' for framework '{framework}'"
        if f"{framework}.optimized_kernel" in config and config[f"{framework}.optimized_kernel"] not in [
            None,
            "cmsis_nn",
        ]:
            RuntimeError(f"There is already a optimized_kernel selected for framework '{framework}'")
        else:
            config[f"{framework}.optimized_kernel"] = "cmsis_nn"
        libs = config.get(f"{framework}.optimized_kernel_libs", [])
        libs.append(self.muriscvnn_lib)
        incs = config.get(f"{framework}.optimized_kernel_inc_dirs", [])
        incs.append(self.muriscvnn_inc_dir)
        config[f"{framework}.optimized_kernel_libs"] = libs
        config[f"{framework}.optimized_kernel_inc_dirs"] = incs

    def get_required_cache_flags(self):
        ret = {}

        ret["tflmc.exe"] = ["muriscvnn"]
        return ret


@register_feature("cmsisnn")
class Cmsisnn(SetupFeature, FrameworkFeature):
    """CMSIS-NN kernels for TFLite Micro/TVM"""

    REQUIRED = ["cmsisnn.lib", "cmsisnn.dir"]

    def __init__(self, config=None):
        super().__init__("cmsisnn", config=config)

    @property
    def cmsisnn_lib(self):
        return str(self.config["cmsisnn.lib"])

    @property
    def cmsisnn_dir(self):
        return str(self.config["cmsisnn.dir"])

    def add_framework_config(self, framework, config):
        assert framework == "tflite", f"Unsupported feature '{self.name}' for framework '{framework}'"
        if f"{framework}.optimized_kernel" in config and config[f"{framework}.optimized_kernel"] not in [
            None,
            "cmsis_nn",
        ]:
            RuntimeError(f"There is already a optimized_kernel selected for framework '{framework}'")
        else:
            config[f"{framework}.optimized_kernel"] = "cmsis_nn"
        libs = config.get(f"{framework}.optimized_kernel_libs", [])
        libs.append(self.cmsisnn_lib)
        incs = config.get(f"{framework}.optimized_kernel_inc_dirs", [])
        include_dirs = [
            self.cmsisnn_dir,
            str(Path(self.cmsisnn_dir) / "CMSIS" / "Core" / "Include"),
            str(Path(self.cmsisnn_dir) / "CMSIS" / "NN" / "Include"),
            str(Path(self.cmsisnn_dir) / "CMSIS" / "DSP" / "Include"),
        ]
        incs.extend(include_dirs)
        config[f"{framework}.optimized_kernel_libs"] = libs
        config[f"{framework}.optimized_kernel_inc_dirs"] = incs

    def get_required_cache_flags(self):
        ret = {}
        ret["tflmc.exe"] = ["cmsisnn"]
        return ret


@register_feature("cmsisnnbyoc")
class CmsisnnByoc(SetupFeature, FrameworkFeature, BackendFeature):
    """CMSIS-NN kernels for TVM using BYOC wrappers."""

    REQUIRED = ["cmsisnn.lib", "cmsisnn.dir"]

    def __init__(self, config=None):
        super().__init__("cmsisnnbyoc", config=config)

    @property
    def cmsisnn_lib(self):
        return str(self.config["cmsisnn.lib"])

    @property
    def cmsisnn_dir(self):
        return str(self.config["cmsisnn.dir"])

    def get_framework_config(self, framework):
        assert framework == "tvm", f"Unsupported feature '{self.name}' for framework '{framework}'"
        include_dirs = [
            self.cmsisnn_dir,
            str(Path(self.cmsisnn_dir) / "CMSIS" / "Core" / "Include"),
            str(Path(self.cmsisnn_dir) / "CMSIS" / "NN" / "Include"),
            str(Path(self.cmsisnn_dir) / "CMSIS" / "DSP" / "Include"),
        ]
        return {
            f"{framework}.extra_libs": [self.cmsisnn_lib],
            f"{framework}.extra_incs": include_dirs,
        }

    def add_backend_config(self, backend, config):
        assert backend in [
            "tvmaot",
            "tvmrt",
            "tvmcg",
        ], f"Unsupported feature '{self.name}' for backend '{backend}'"
        extras = config.get(f"{backend}.extra_kernel", [])
        if "cmsis-nn" not in extras:
            extras[f"{backend}.extra_kernel"].append("cmsis-nn")
        config[f"{backend}.extra_kernel"] = extras

    def get_required_cache_flags(self):
        ret = {}
        ret["tvm.build_dir"] = ["cmsisnn"]
        return ret


# @before_feature("muriscvnn")  # TODO: implment something like this
@register_feature("vext")
# class Vext(SetupFeature, TargetFeature, PlatformFeature):
class Vext(SetupFeature, TargetFeature):
    """MuriscvNN CMSIS-NN wrappers for TFLite Micro"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "vlen": 64,  # TODO; define reasonable default? (Or put defaults in target and overwrite of not None)
    }

    REQUIRED = []

    def __init__(self, config=None):
        super().__init__("vext", config=config)

    @property
    def vlen(self):
        return int(self.config["vlen"])

    def get_target_config(self, target):
        # TODO: enforce llvm toolchain using add_compile_config and CompileFeature?
        assert target in ["spike", "ovpsim"]
        assert is_power_of_two(self.vlen)
        return {
            f"{target}.enable_vext": True,
            f"{target}.vlen": self.vlen,
        }

    # It would be great if we could enforce an llvm toolchain here
    # def add_compile_config(self, config):
    #     # TODO: enforce llvm toolchain using add_compile_config and CompileFeature?
    #     if "mlif.toolchain" in config:
    #         assert "mlif.toolchain" == "llvm", "Vext requires LLVM target sw"
    #     else:
    #         config["mlif.toolchain"] = "llvm"

    def get_required_cache_flags(self):
        return {
            "muriscvnn.lib": ["vext"],
            "muriscvnn.inc_dir": ["vext"],
            "tflmc.exe": ["vext"],
        }


@register_feature("debug")
class Debug(SetupFeature, PlatformFeature):
    """Enable debugging ability of target software."""

    def __init__(self, config=None):
        super().__init__("debug", config=config)

    def get_required_cache_flags(self):
        return {} if self.enabled else {}  # TODO: remove?

    def get_platform_config(self, platform):
        return {f"{platform}.debug": self.enabled}


@register_feature("gdbserver")
class GdbServer(TargetFeature):
    """Start debugging session for target software using gdbserver."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "attach": None,
        "port": None,
    }

    def __init__(self, config=None):
        super().__init__("gdbserver", config=config)

    @property
    def attach(self):
        # TODO: implement get_bool_or_none?
        return bool(self.config["attach"]) if self.config["attach"] is not None else None

    @property
    def port(self):
        return int(self.config["port"]) if self.config["port"] is not None else None

    def get_target_config(self, target):
        assert target in ["host_x86", "etiss_pulpino"]
        return filter_none(
            {
                f"{target}.gdbserver_enable": self.enabled,
                f"{target}.gdbserver_attach": self.attach,
                f"{target}.gdbserver_port": self.port,
            }
        )


@register_feature("etissdbg")
class ETISSDebug(SetupFeature, TargetFeature):
    """Debug ETISS internals."""

    def __init__(self, config=None):
        super().__init__("etissdbg", config=config)

    def get_required_cache_flags(self):
        return {"etiss.install_dir": ["debug"], "etissvp.script": ["debug"]} if self.enabled else {}

    def get_target_config(self, target):
        assert target in ["etiss_pulpino"]
        return {"etiss_pulpino.debug_etiss": self.enabled}


@register_feature("trace")
class Trace(TargetFeature):
    """Enable tracing of all memory accesses in ETISS."""

    def __init__(self, config=None):
        super().__init__("etissdbg", config=config)

    def get_target_config(self, target):
        assert target in ["etiss_pulpino"]
        return {"etiss_pulpino.trace_memory": self.enabled}


@register_feature("unpacked_api")
class UnpackedApi(BackendFeature):  # TODO: should this be a feature or config only?
    """Use unpacked interface api for TVMAOT backend to reduce stack usage."""

    def __init__(self, config=None):
        super().__init__("unpacked_api", config=config)

    def get_backend_config(self, backend):
        assert backend in ["tvmaot"], f"Unsupported feature '{self.name}' for backend '{backend}'"
        return {f"{backend}.unpacked_api": self.enabled}


@register_feature("packed")
class Packed(FrameworkFeature, FrontendFeature, BackendFeature, SetupFeature):
    """Sub-8-bit and sparsity feature for TFLite Micro kernels."""

    def __init__(self, config=None):
        super().__init__("packed", config=config)

    def get_framework_config(self, framework):
        raise NotImplementedError

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name} for frontend '{frontend}''"
        return {f"{frontend}.use_packed_weights": self.enabled}

    def get_backend_config(self, backend):
        raise NotImplementedError

    def get_required_cache_flags(self):
        return {"tflmc.exe": ["packed"]}


@register_feature("packing")
class Packing(FrontendFeature):
    """Sub-8-bit and sparse weight packing for TFLite Frontend."""

    def __init__(self, config=None):
        super().__init__("packing", config=config)

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name} for frontend '{frontend}''"
        raise NotImplementedError
        return {f"{frontend}.pack_weights": self.enabled}


@register_feature("memplan")
class Memplan(FrameworkFeature):
    """Custom TVM memory planning feature by (@rafzi)"""

    def __init__(self, config=None):
        super().__init__("memplan", config=config)

    def get_framework_config(self, framework):
        assert framework in ["tvm"], f"Usupported fetaure '{self.name}' for framework '{framework}'"
        raise NotImplementedError
        return {"tvm.memplan_enable": self.enabled}


@register_feature("usmp")
class Usmp(BackendFeature):
    """Unified Static Memory Planning algorithm integrated in TVM"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "algorithm": "greedy_by_conflicts",  # options: greedy_by_conflicts, greedy_by_size, hill_climb
    }

    def __init__(self, config=None):
        super().__init__("usmp", config=config)

    @property
    def algorithm(self):
        return str(self.config["algorithm"])

    def add_backend_config(self, backend, config):
        assert backend in ["tvmaot"], f"Usupported fetaure '{self.name}' for backend '{backend}'"
        if f"{backend}.extra_pass_config" in config:
            tmp = config[f"{backend}.extra_pass_config"]
        elif "extra_pass_config" in config:
            tmp = config["extra_pass_config"]
        else:
            tmp = {}
        tmp["tir.usmp.enable"] = self.enabled
        tmp["tir.usmp.algorithm"] = self.algorithm
        config.update({f"{backend}.extra_pass_config": tmp})

    # -> enable this via backend


@register_feature("fusetile")
class Fusetile(FrameworkFeature):  # TODO: rename to MOIOPT?
    """WIP TVM feature by (@rafzi)"""

    def __init__(self, config=None):
        super().__init__("fusetile", config=config)

    def get_framework_config(self, framework):
        assert framework in ["tvm"], f"Usupported fetaure '{self.name}' for framework '{framework}'"
        raise NotImplementedError
        return {"tvm.fusetile_enable": self.enabled}

    # -> enable this via backend


@register_feature("visualize")
class Visualize(BackendFeature):
    """Visualize TVM relay models."""

    # Bokeh backend has additional python requirements: graphviz, pydot, bokeh >= 2.3.1
    # TODO: add tflite visualizer? (Frontend)

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "mode": "cli",  # Alternative: bokeh
    }

    def __init__(self, config=None):
        super().__init__("visualize", config=config)

    @property
    def mode(self):
        value = self.config["mode"] if "mode" in self.config else None
        if value:
            assert value.lower() in ["cli", "bokeh"]
        return value

    def get_backend_config(self, backend):
        assert backend in TVM_BACKENDS, f"Unsupported feature '{self.name}' for backend '{backend}'"  # TODO: undefined!
        return NotImplementedError
        return filter_none(
            {
                f"{backend}.visualize_enable": self.enabled,
                f"{backend}.visualize_mode": self.mode,
            }
        )


@register_feature("autotuned")
class Autotuned(BackendFeature):
    """Use existing TVM autotuning logs in backend."""

    # TODO: FronendFeature to collect tuning logs or will we store them somewhere else?

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "results_file": None,
    }

    def __init__(self, config=None):
        super().__init__("autotuned", config=config)

    @property
    def results_file(self):
        return self.config["results_file"] if "results_file" in self.config else None

    def get_backend_config(self, backend):
        assert backend in ["tvmaot", "tvmcg", "tvmrt"]  # TODO: backend in TVM_BACKENDS
        # TODO: error handling her eor on backend?
        return filter_none(
            {
                f"{backend}.use_tuning_results": self.enabled,
                f"{backend}.autotuning_results_file": self.results_file,
            }
        )


@register_feature("autotune")
class Autotune(BackendFeature, RunFeature):
    """Use the TVM autotuner inside the backend to generate tuning logs."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "results_file": None,
        "append": None,
        "tuner": None,
        "trial": None,
        "early_stopping": None,
        "num_workers": None,
        "max_parallel": None,
        "use_rpc": None,
        "timeout": None,
        # All None to use the defaults defined in the backend instead
    }

    def __init__(self, config=None):
        super().__init__("autotune", config=config)

    @property
    def results_file(self):
        return self.config["results_file"] if "results_file" in self.config else None

    @property
    def append(self):
        return self.config["append"] if "append" in self.config else None

    @property
    def tuner(self):
        return self.config["tuner"] if "tuner" in self.config else None

    @property
    def trials(self):
        return self.config["trials"] if "trials" in self.config else None

    @property
    def early_stopping(self):
        return self.config["early_stopping"] if "early_stopping" in self.config else None

    @property
    def num_workers(self):
        return self.config["num_workers"] if "num_workers" in self.config else None

    @property
    def max_parallel(self):
        return self.config["max_parallel"] if "max_parallel" in self.config else None

    @property
    def use_rpc(self):
        return self.config["use_rpc"] if "use_rpc" in self.config else None

    @property
    def timeout(self):
        return self.config["timeout"] if "timeout" in self.config else None

    def get_backend_config(self, backend):
        assert backend in ["tvmaot", "tvmcg", "tvmrt"]  # TODO: backend in TVM_BACKENDS
        # TODO: figure out a default path automatically
        return filter_none(
            {
                f"{backend}.autotuning_enable": self.enabled,
                # f"{backend}.autotuning_use_tuned": self.enabled,  # Should Autotuning ==> Autotuned?
                f"{backend}.autotuning_results_file": self.results_file,
                f"{backend}.autotuning_append": self.append,
                f"{backend}.autotuning_tuner": self.tuner,
                f"{backend}.autotuning_trials": self.trials,
                f"{backend}.autotuning_early_stopping": self.early_stopping,
                f"{backend}.autotuning_num_workers": self.num_workers,
                f"{backend}.autotuning_max_parallel": self.max_parallel,
                f"{backend}.autotuning_use_rpc": self.use_rpc,
                f"{backend}.autotuning_timeout": self.timeout,
            }
        )

    def get_run_config(self):
        return {"run.tune_enabled": self.enabled}


@register_feature("disable_legalize")
class DebugArena(BackendFeature, PlatformFeature):
    """Enable verbose printing of arena usage for debugging."""

    def __init__(self, config=None):
        super().__init__("debug_arena", config=config)

    def get_backend_config(self, backend):
        assert backend in [
            "tvmaot",
            "tvmcg",
            "tvmrt",
        ], f"Unsupported feature '{self.name}' for backend '{backend}'"
        # TODO: TFLM also compatible?
        return {f"{backend}.debug_arena": self.enabled}

    # def get_platform_config(self):
    #    return {"mlif.debug_arena": True}

    def get_cmake_args(self):
        val = "ON" if self.enabled else "OFF"
        return [f"-DDEBUG_ARENA={val}"]


@register_feature("validate")
class Validate(FrontendFeature, PlatformFeature):
    """Enable validaton of inout and output tensors."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "allow_missing": True,
    }

    def __init__(self, config=None):
        super().__init__("validate", config=config)

    @property
    def allow_missing(self):
        return bool(self.config["allow_missing"])

    def get_frontend_config(self, frontend):
        if not self.allow_missing:
            raise NotImplementedError
        return {f"{frontend}.use_inout_data": True}

    def get_platform_config(self, platform):
        assert platform == "mlif", f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {f"{platform}.ignore_data": False}

    # def get_cmake_args(self):
    #     pass


@register_feature("muriscvnn")
class Muriscvnn(SetupFeature, FrameworkFeature):
    """MuriscvNN CMSIS-NN wrappers for TFLite Micro"""

    REQUIRED = ["muriscvnn.lib", "muriscvnn.inc_dir"]

    def __init__(self, config=None):
        super().__init__("muriscvnn", config=config)

    @property
    def muriscvnn_lib(self):
        return str(self.config["muriscvnn.lib"])

    @property
    def muriscvnn_inc_dir(self):
        return str(self.config["muriscvnn.inc_dir"])

    def add_framework_config(self, framework, config):
        assert framework == "tflite", f"Unsupported feature '{self.name}' for framework '{framework}'"
        if f"{framework}.optimized_kernel" in config and config[f"{framework}.optimized_kernel"] not in [
            None,
            "cmsis_nn",
        ]:
            RuntimeError(f"There is already a optimized_kernel selected for framework '{framework}'")
        else:
            config[f"{framework}.optimized_kernel"] = "cmsis_nn"
        libs = config.get(f"{framework}.optimized_kernel_libs", [])
        libs.append(self.muriscvnn_lib)
        incs = config.get(f"{framework}.optimized_kernel_inc_dirs", [])
        incs.append(self.muriscvnn_inc_dir)
        config[f"{framework}.optimized_kernel_libs"] = libs
        config[f"{framework}.optimized_kernel_inc_dirs"] = incs

    def get_required_cache_flags(self):
        ret = {}

        ret["tflmc.exe"] = ["muriscvnn"]
        return ret


@register_feature("cmsisnn")
class Cmsisnn(SetupFeature, FrameworkFeature):
    """CMSIS-NN kernels for TFLite Micro/TVM"""

    REQUIRED = ["cmsisnn.lib", "cmsisnn.dir"]

    def __init__(self, config=None):
        super().__init__("cmsisnn", config=config)

    @property
    def cmsisnn_lib(self):
        return str(self.config["cmsisnn.lib"])

    @property
    def cmsisnn_dir(self):
        return str(self.config["cmsisnn.dir"])

    def add_framework_config(self, framework, config):
        assert framework == "tflite", f"Unsupported feature '{self.name}' for framework '{framework}'"
        if f"{framework}.optimized_kernel" in config and config[f"{framework}.optimized_kernel"] not in [
            None,
            "cmsis_nn",
        ]:
            RuntimeError(f"There is already a optimized_kernel selected for framework '{framework}'")
        else:
            config[f"{framework}.optimized_kernel"] = "cmsis_nn"
        libs = config.get(f"{framework}.optimized_kernel_libs", [])
        libs.append(self.cmsisnn_lib)
        incs = config.get(f"{framework}.optimized_kernel_inc_dirs", [])
        include_dirs = [
            self.cmsisnn_dir,
            str(Path(self.cmsisnn_dir) / "CMSIS" / "Core" / "Include"),
            str(Path(self.cmsisnn_dir) / "CMSIS" / "NN" / "Include"),
            str(Path(self.cmsisnn_dir) / "CMSIS" / "DSP" / "Include"),
        ]
        incs.extend(include_dirs)
        config[f"{framework}.optimized_kernel_libs"] = libs
        config[f"{framework}.optimized_kernel_inc_dirs"] = incs

    def get_required_cache_flags(self):
        ret = {}
        ret["tflmc.exe"] = ["cmsisnn"]
        return ret


@register_feature("cmsisnnbyoc")
class CmsisnnByoc(SetupFeature, FrameworkFeature, BackendFeature):
    """CMSIS-NN kernels for TVM using BYOC wrappers."""

    REQUIRED = ["cmsisnn.lib", "cmsisnn.dir"]

    def __init__(self, config=None):
        super().__init__("cmsisnnbyoc", config=config)

    @property
    def cmsisnn_lib(self):
        return str(self.config["cmsisnn.lib"])

    @property
    def cmsisnn_dir(self):
        return str(self.config["cmsisnn.dir"])

    def get_framework_config(self, framework):
        assert framework == "tvm", f"Unsupported feature '{self.name}' for framework '{framework}'"
        include_dirs = [
            self.cmsisnn_dir,
            str(Path(self.cmsisnn_dir) / "CMSIS" / "Core" / "Include"),
            str(Path(self.cmsisnn_dir) / "CMSIS" / "NN" / "Include"),
            str(Path(self.cmsisnn_dir) / "CMSIS" / "DSP" / "Include"),
        ]
        return {
            f"{framework}.extra_libs": [self.cmsisnn_lib],
            f"{framework}.extra_incs": include_dirs,
        }

    def add_backend_config(self, backend, config):
        assert backend in [
            "tvmaot",
            "tvmrt",
            "tvmcg",
        ], f"Unsupported feature '{self.name}' for backend '{backend}'"
        extras = config.get(f"{backend}.extra_kernel", [])
        if "cmsis-nn" not in extras:
            extras[f"{backend}.extra_kernel"].append("cmsis-nn")
        config[f"{backend}.extra_kernel"] = extras

    def get_required_cache_flags(self):
        ret = {}
        ret["tvm.build_dir"] = ["cmsisnn"]
        return ret


# @before_feature("muriscvnn")  # TODO: implment something like this
@register_feature("vext")
# class Vext(SetupFeature, TargetFeature, PlatformFeature):
class Vext(SetupFeature, TargetFeature):
    """MuriscvNN CMSIS-NN wrappers for TFLite Micro"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "vlen": 64,  # TODO; define reasonable default? (Or put defaults in target and overwrite of not None)
    }

    REQUIRED = []

    def __init__(self, config=None):
        super().__init__("vext", config=config)

    @property
    def vlen(self):
        return int(self.config["vlen"])

    def get_target_config(self, target):
        # TODO: enforce llvm toolchain using add_compile_config and CompileFeature?
        assert target in ["spike", "ovpsim"]
        assert is_power_of_two(self.vlen)
        return {
            f"{target}.enable_vext": True,
            f"{target}.vlen": self.vlen,
        }

    # It would be great if we could enforce an llvm toolchain here
    # def add_compile_config(self, config):
    #     # TODO: enforce llvm toolchain using add_compile_config and CompileFeature?
    #     if "mlif.toolchain" in config:
    #         assert "mlif.toolchain" == "llvm", "Vext requires LLVM target sw"
    #     else:
    #         config["mlif.toolchain"] = "llvm"

    def get_required_cache_flags(self):
        return {
            "muriscvnn.lib": ["vext"],
            "muriscvnn.inc_dir": ["vext"],
            "tflmc.exe": ["vext"],
        }


@register_feature("debug")
class Debug(SetupFeature, PlatformFeature):
    """Enable debugging ability of target software."""

    def __init__(self, config=None):
        super().__init__("debug", config=config)

    def get_required_cache_flags(self):
        return {} if self.enabled else {}  # TODO: remove?

    def get_platform_config(self, platform):
        return {f"{platform}.debug": self.enabled}


@register_feature("gdbserver")
class GdbServer(TargetFeature):
    """Start debugging session for target software using gdbserver."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "attach": None,
        "port": None,
    }

    def __init__(self, config=None):
        super().__init__("gdbserver", config=config)

    @property
    def attach(self):
        # TODO: implement get_bool_or_none?
        return bool(self.config["attach"]) if self.config["attach"] is not None else None

    @property
    def port(self):
        return int(self.config["port"]) if self.config["port"] is not None else None

    def get_target_config(self, target):
        assert target in ["host_x86", "etiss_pulpino"]
        return filter_none(
            {
                f"{target}.gdbserver_enable": self.enabled,
                f"{target}.gdbserver_attach": self.attach,
                f"{target}.gdbserver_port": self.port,
            }
        )


@register_feature("etissdbg")
class ETISSDebug(SetupFeature, TargetFeature):
    """Debug ETISS internals."""

    def __init__(self, config=None):
        super().__init__("etissdbg", config=config)

    def get_required_cache_flags(self):
        return {"etiss.install_dir": ["debug"], "etissvp.script": ["debug"]} if self.enabled else {}

    def get_target_config(self, target):
        assert target in ["etiss_pulpino"]
        return {"etiss_pulpino.debug_etiss": self.enabled}


@register_feature("trace")
class Trace(TargetFeature):
    """Enable tracing of all memory accesses in ETISS."""

    def __init__(self, config=None):
        super().__init__("etissdbg", config=config)

    def get_target_config(self, target):
        assert target in ["etiss_pulpino"]
        return {"etiss_pulpino.trace_memory": self.enabled}


@register_feature("unpacked_api")
class UnpackedApi(BackendFeature):  # TODO: should this be a feature or config only?
    """Use unpacked interface api for TVMAOT backend to reduce stack usage."""

    def __init__(self, config=None):
        super().__init__("unpacked_api", config=config)

    def get_backend_config(self, backend):
        assert backend in ["tvmaot"], f"Unsupported feature '{self.name}' for backend '{backend}'"
        return {f"{backend}.unpacked_api": self.enabled}


@register_feature("packed")
class Packed(FrameworkFeature, FrontendFeature, BackendFeature, SetupFeature, PlatformFeature):
    """Sub-8-bit and sparsity feature for TFLite Micro kernels."""

    def __init__(self, config=None):
        super().__init__("packed", config=config)

    def get_framework_config(self, framework):
        raise NotImplementedError

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name} for frontend '{frontend}''"
        return {f"{frontend}.use_packed_weights": self.enabled}

    def get_backend_config(self, backend):
        raise NotImplementedError

    def get_required_cache_flags(self):
        return {"tflmc.exe": ["packed"]}

    def get_cmake_args(self):
        val = "ON" if self.enabled else "OFF"
        return [f"-DDEBUG_ARENA={val}"]


@register_feature("packing")
class Packing(FrontendFeature):
    """Sub-8-bit and sparse weight packing for TFLite Frontend."""

    def __init__(self, config=None):
        super().__init__("packing", config=config)

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name} for frontend '{frontend}''"
        raise NotImplementedError
        return {f"{frontend}.pack_weights": self.enabled}


@register_feature("fallback")
class Fallback(FrameworkFeature, PlatformFeature):
    """(Unimplemented) TFLite Fallback for unsupported and custom operators in TVM."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "config_file": None,
    }

    def __init__(self, config=None):
        super().__init__("fallback", config=config)

    @property
    def config_file(self):
        return str(self.config["config_file"]) if "config_file" in self.config else None

    def get_framework_config(self, framework):
        assert framework in ["tvm"], f"Usupported fetaure '{self.name}' for framework '{framework}'"
        raise NotImplementedError
        return filter_none(
            {
                f"{framework}.fallback_enable": self.enabled,
                f"{framework}.fallback_config_file": self.config_file,
            }
        )

    # -> hard to model..., preprocess for tflmc?


@register_feature("memplan")
class Memplan(FrameworkFeature):
    """Custom TVM memory planning feature by (@rafzi)"""

    def __init__(self, config=None):
        super().__init__("memplan", config=config)

    def get_framework_config(self, framework):
        assert framework in ["tvm"], f"Usupported fetaure '{self.name}' for framework '{framework}'"
        return {"tvm.memplan_enable": self.enabled}


@register_feature("usmp")
class Usmp(BackendFeature):
    """Unified Static Memory Planning algorithm integrated in TVM"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "algorithm": "greedy_by_conflicts",  # options: greedy_by_conflicts, greedy_by_size, hill_climb
    }

    def __init__(self, config=None):
        super().__init__("usmp", config=config)

    @property
    def algorithm(self):
        return str(self.config["algorithm"])

    def add_backend_config(self, backend, config):
        assert backend in ["tvmaot"], f"Usupported fetaure '{self.name}' for backend '{backend}'"
        if f"{backend}.extra_pass_config" in config:
            tmp = config[f"{backend}.extra_pass_config"]
        elif "extra_pass_config" in config:
            tmp = config["extra_pass_config"]
        else:
            tmp = {}
        tmp["tir.usmp.enable"] = self.enabled
        tmp["tir.usmp.algorithm"] = self.algorithm
        config.update({f"{backend}.extra_pass_config": tmp})

    # -> enable this via backend


@register_feature("fusetile")
class Fusetile(FrameworkFeature):
    """WIP TVM feature by (@rafzi)"""

    def __init__(self, config=None):
        super().__init__("fusetile", config=config)

    def get_framework_config(self, framework):
        assert framework in ["tvm"], f"Usupported fetaure '{self.name}' for framework '{framework}'"
        return {"tvm.fusetile_enable": self.enabled}

    # -> enable this via backend


@register_feature("visualize")
class Visualize(BackendFeature):
    """Visualize TVM relay models."""

    # Bokeh backend has additional python requirements: graphviz, pydot, bokeh >= 2.3.1
    # TODO: add tflite visualizer? (Frontend)

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "mode": "cli",  # Alternative: bokeh
    }

    def __init__(self, config=None):
        super().__init__("visualize", config=config)

    @property
    def mode(self):
        value = self.config["mode"] if "mode" in self.config else None
        if value:
            assert value.lower() in ["cli", "bokeh"]
        return value

    def get_backend_config(self, backend):
        assert backend in TVM_BACKENDS, f"Unsupported feature '{self.name}' for backend '{backend}'"  # TODO: undefined!
        return NotImplementedError
        return filter_none(
            {
                f"{backend}.visualize_enable": self.enabled,
                f"{backend}.visualize_mode": self.mode,
            }
        )


@register_feature("autotuned")
class Autotuned(BackendFeature):
    """Use existing TVM autotuning logs in backend."""

    # TODO: FronendFeature to collect tuning logs or will we store them somewhere else?

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "results_file": None,
    }

    def __init__(self, config=None):
        super().__init__("autotuned", config=config)

    @property
    def results_file(self):
        return self.config["results_file"] if "results_file" in self.config else None

    def get_backend_config(self, backend):
        assert backend in ["tvmaot", "tvmcg", "tvmrt"]  # TODO: backend in TVM_BACKENDS
        # TODO: error handling her eor on backend?
        return filter_none(
            {
                f"{backend}.use_tuning_results": self.enabled,
                f"{backend}.autotuning_results_file": self.results_file,
            }
        )


@register_feature("autotune")
class Autotune(BackendFeature, RunFeature):
    """Use the TVM autotuner inside the backend to generate tuning logs."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "results_file": None,
        "append": None,
        "tuner": None,
        "trial": None,
        "early_stopping": None,
        "num_workers": None,
        "max_parallel": None,
        "use_rpc": None,
        "timeout": None,
        # All None to use the defaults defined in the backend instead
    }

    def __init__(self, config=None):
        super().__init__("autotune", config=config)

    @property
    def results_file(self):
        return self.config["results_file"] if "results_file" in self.config else None

    @property
    def append(self):
        return self.config["append"] if "append" in self.config else None

    @property
    def tuner(self):
        return self.config["tuner"] if "tuner" in self.config else None

    @property
    def trials(self):
        return self.config["trials"] if "trials" in self.config else None

    @property
    def early_stopping(self):
        return self.config["early_stopping"] if "early_stopping" in self.config else None

    @property
    def num_workers(self):
        return self.config["num_workers"] if "num_workers" in self.config else None

    @property
    def max_parallel(self):
        return self.config["max_parallel"] if "max_parallel" in self.config else None

    @property
    def use_rpc(self):
        return self.config["use_rpc"] if "use_rpc" in self.config else None

    @property
    def timeout(self):
        return self.config["timeout"] if "timeout" in self.config else None

    def get_backend_config(self, backend):
        assert backend in ["tvmaot", "tvmcg", "tvmrt"]  # TODO: backend in TVM_BACKENDS
        # TODO: figure out a default path automatically
        return filter_none(
            {
                f"{backend}.autotuning_enable": self.enabled,
                # f"{backend}.autotuning_use_tuned": self.enabled,  # Should Autotuning ==> Autotuned?
                f"{backend}.autotuning_results_file": self.results_file,
                f"{backend}.autotuning_append": self.append,
                f"{backend}.autotuning_tuner": self.tuner,
                f"{backend}.autotuning_trials": self.trials,
                f"{backend}.autotuning_early_stopping": self.early_stopping,
                f"{backend}.autotuning_num_workers": self.num_workers,
                f"{backend}.autotuning_max_parallel": self.max_parallel,
                f"{backend}.autotuning_use_rpc": self.use_rpc,
                f"{backend}.autotuning_timeout": self.timeout,
            }
        )

    def get_run_config(self):
        return {"run.tune_enabled": self.enabled}


@register_feature("disable_legalize")
class DisableLegalize(BackendFeature, SetupFeature):
    """Enable transformation to reduces sizes of intermediate buffers by skipping legalization passes."""

    REQUIRED = ["tvm_extensions.wrapper"]

    def __init__(self, config=None):
        super().__init__("disable_legalize", config=config)

    @property
    def tvm_extensions_wrapper(self):
        return self.config["tvm_extensions.wrapper"]

    def add_backend_config(self, backend, config):
        assert backend in [
            "tvmaot",
            "tvmcg",
            "tvmrt",
        ], f"Unsupported feature '{self.name}' for backend '{backend}'"
        if f"{backend}.tvmc_extra_args" in config:
            config[f"{backend}.tvmc_extra_args"].append("--disable-legalize")
        else:
            config[f"{backend}.tvmc_extra_args"] = ["--disable-legalize"]
        if f"{backend}.tvmc_custom_script" in config:
            assert config[f"{backend}.tvmc_custom_script"] is None or str(
                config[f"{backend}.tvmc_custom_script"]
            ) == str(
                self.tvm_extensions_src
            ), f"{backend}.tvmc_custom_script is already set. Can't enable feature: {self.name}"
        config[f"{backend}.tvmc_custom_script"] = self.tvm_extensions_wrapper

    def get_required_cache_flags(self):
        ret = {}

        ret["tvm.pythonpath"] = ["patch"]
        return ret
