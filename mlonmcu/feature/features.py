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

import re
from pathlib import Path

from mlonmcu.utils import is_power_of_two
from mlonmcu.config import str2bool
from mlonmcu.artifact import Artifact, ArtifactFormat
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

# from mlonmcu.flow import SUPPORTED_TVM_BACKENDS
SUPPORTED_TVM_BACKENDS = [
    "tvmaot",
    "tvmrt",
    "tvmcg",
    "tvmllvm",
]  # Workaround for cirvular import until we have a backend registry


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
class DebugArena(BackendFeature):
    """Enable verbose printing of arena usage for debugging."""

    def __init__(self, features=None, config=None):
        super().__init__("debug_arena", features=features, config=config)

    def get_backend_config(self, backend):
        assert backend in [
            "tvmaot",
            "tvmrt",
            "tflmi",
        ], f"Unsupported feature '{self.name}' for backend '{backend}'"
        return {f"{backend}.debug_arena": self.enabled}


@register_feature("validate")
class Validate(FrontendFeature, PlatformFeature):
    """Enable validaton of inout and output tensors."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "allow_missing": True,
        "fail_on_error": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("validate", features=features, config=config)

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
        return filter_none(
            {
                f"{platform}.ignore_data": False,
                f"{platform}.fail_on_error": self.fail_on_error,
            }
        )


@register_feature("muriscvnn")
class Muriscvnn(SetupFeature, FrameworkFeature, PlatformFeature):
    """muRISCV-V NN wrappers for TFLite Micro"""

    REQUIRED = ["muriscvnn.src_dir"]

    def __init__(self, features=None, config=None):
        super().__init__("muriscvnn", features=features, config=config)

    @property
    def muriscvnn_dir(self):
        return str(self.config["muriscvnn.src_dir"])

    def add_framework_config(self, framework, config):
        assert framework == "tflm", f"Unsupported feature '{self.name}' for framework '{framework}'"
        if f"{framework}.optimized_kernel" in config and config[f"{framework}.optimized_kernel"] not in [
            None,
            "cmsis_nn",
        ]:
            RuntimeError(f"There is already a optimized_kernel selected for framework '{framework}'")
        else:
            config[f"{framework}.optimized_kernel"] = "cmsis_nn"

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {
            "MURISCVNN": self.enabled,
            "MURISCVNN_DIR": self.muriscvnn_dir,
        }

    def get_required_cache_flags(self):
        ret = {}

        ret["tflmc.exe"] = ["muriscvnn"]
        return ret


@register_feature("cmsisnn")
class Cmsisnn(SetupFeature, FrameworkFeature, PlatformFeature):
    """CMSIS-NN kernels for TFLite Micro"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
    }

    REQUIRED = ["cmsisnn.dir"]

    def __init__(self, features=None, config=None):
        super().__init__("cmsisnn", features=features, config=config)

    @property
    def cmsisnn_dir(self):
        return str(self.config["cmsisnn.dir"])

    def add_framework_config(self, framework, config):
        assert framework == "tflm", f"Unsupported feature '{self.name}' for framework '{framework}'"
        if f"{framework}.optimized_kernel" in config and config[f"{framework}.optimized_kernel"] not in [
            None,
            "cmsis_nn",
        ]:
            RuntimeError(f"There is already a optimized_kernel selected for framework '{framework}'")
        else:
            config[f"{framework}.optimized_kernel"] = "cmsis_nn"

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {
            "CMSISNN": self.enabled,
            "CMSISNN_DIR": self.cmsisnn_dir,
        }

    def get_required_cache_flags(self):
        ret = {}
        ret["tflmc.exe"] = ["cmsisnn"]
        return ret


@register_feature("cmsisnnbyoc")
class CmsisnnByoc(SetupFeature, BackendFeature, PlatformFeature):
    """CMSIS-NN kernels for TVM using BYOC wrappers."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "mcpu": None,  # mve: cortex-m55, dsp: cortex-m4, cortex-m7, cortex-m33, cortex-m35p
    }

    REQUIRED = ["cmsisnn.dir"]

    def __init__(self, features=None, config=None):
        super().__init__("cmsisnnbyoc", features=features, config=config)

    @property
    def cmsisnn_dir(self):
        return str(self.config["cmsisnn.dir"])

    @property
    def mcpu(self):
        return self.config["mcpu"]

    def add_backend_config(self, backend, config):
        assert backend in SUPPORTED_TVM_BACKENDS, f"Unsupported feature '{self.name}' for backend '{backend}'"
        extras = config.get(f"{backend}.extra_target", [])
        if "cmsis-nn" not in extras:
            if isinstance(extras, str):
                extras = [extras]
            extras.append("cmsis-nn")
        config[f"{backend}.extra_target"] = extras
        if self.mcpu:
            # Ideally cmsisnnbyoc would have a mvei/dsp feature which could be used to set this automatically
            config[f"{backend}.extra_target_mcpu"] = self.mcpu

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {
            "CMSISNN": self.enabled,
            "CMSISNN_DIR": self.cmsisnn_dir,
        }

    def get_required_cache_flags(self):
        ret = {}
        ret["tvm.build_dir"] = ["cmsisnn"]
        return ret


@register_feature("muriscvnnbyoc")
class MuriscvnnByoc(SetupFeature, BackendFeature, PlatformFeature):
    """MuRiscvNN kernels for TVM using BYOC wrappers."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "mcpu": None,  # mve: cortex-m55, dsp: cortex-m4, cortex-m7, cortex-m33, cortex-m35p
    }

    REQUIRED = ["muriscvnn.src_dir"]

    def __init__(self, features=None, config=None):
        super().__init__("muriscvnnbyoc", features=features, config=config)

    @property
    def muriscvnn_dir(self):
        return str(self.config["muriscvnn.src_dir"])

    @property
    def mcpu(self):
        return self.config["mcpu"]

    def add_backend_config(self, backend, config):
        assert backend in SUPPORTED_TVM_BACKENDS, f"Unsupported feature '{self.name}' for backend '{backend}'"
        extras = config.get(f"{backend}.extra_target", [])
        if "cmsis-nn" not in extras:
            if isinstance(extras, str):
                extras = [extras]
            extras.append("cmsis-nn")
        config[f"{backend}.extra_target"] = extras
        if self.mcpu:
            # Ideally muriscvnnbyoc would have a vext/pext feature which could be used to set this automatically
            config[f"{backend}.extra_target_mcpu"] = self.mcpu

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {
            "MURISCVNN": self.enabled,
            "MURISCVNN_DIR": self.muriscvnn_dir,
        }

    def get_required_cache_flags(self):
        ret = {}
        ret["tvm.build_dir"] = ["cmsisnn"]
        return ret


# @before_feature("muriscvnn")  # TODO: implement something like this
@register_feature("vext")
class Vext(SetupFeature, TargetFeature, PlatformFeature):
    """Enable vector extension for supported RISC-V targets"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "vlen": 64,  # TODO; define reasonable default? (Or put defaults in target and overwrite of not None)
    }

    REQUIRED = []

    def __init__(self, features=None, config=None):
        super().__init__("vext", features=features, config=config)

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

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {"RISCV_VEXT": self.enabled}

    def get_required_cache_flags(self):
        return {
            "muriscvnn.lib": ["vext"],
            "tflmc.exe": ["vext"],
            "riscv_gcc.install_dir": ["vext"],
        }


@register_feature("pext")
class Pext(SetupFeature, TargetFeature):
    """Enable packed SIMD extension for supported RISC-V targets"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
    }

    REQUIRED = []

    def __init__(self, features=None, config=None):
        super().__init__("pext", features=features, config=config)

    def get_target_config(self, target):
        assert target in ["spike", "ovpsim"]  # TODO: add etiss in the future
        return {
            f"{target}.enable_pext": True,  # Handle via arch characters in the future
        }

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {"RISCV_PEXT": self.enabled}

    def get_required_cache_flags(self):
        # These will be merged automatically with existing ones
        return {
            "muriscvnn.lib": ["pext"],
            "tflmc.exe": ["pext"],
            "riscv_gcc.install_dir": ["pext"],
        }


@register_feature("debug")
class Debug(SetupFeature, PlatformFeature):
    """Enable debugging ability of target software."""

    def __init__(self, features=None, config=None):
        super().__init__("debug", features=features, config=config)

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

    def __init__(self, features=None, config=None):
        super().__init__("gdbserver", features=features, config=config)

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

    def __init__(self, features=None, config=None):
        super().__init__("etissdbg", features=features, config=config)

    def get_required_cache_flags(self):
        return {"etiss.install_dir": ["dbg"], "etissvp.script": ["dbg"]} if self.enabled else {}

    def get_target_config(self, target):
        assert target in ["etiss_pulpino"]
        return {"etiss_pulpino.debug_etiss": self.enabled}


@register_feature("trace")
class Trace(TargetFeature):
    """Enable tracing of all memory accesses in ETISS."""

    def __init__(self, features=None, config=None):
        super().__init__("trace", features=features, config=config)

    def get_target_config(self, target):
        assert target in ["etiss_pulpino"]
        return {"etiss_pulpino.trace_memory": self.enabled}


@register_feature("unpacked_api")
class UnpackedApi(BackendFeature):  # TODO: should this be a feature or config only?
    """Use unpacked interface api for TVMAOT backend to reduce stack usage."""

    def __init__(self, features=None, config=None):
        super().__init__("unpacked_api", features=features, config=config)

    def get_backend_config(self, backend):
        assert backend in ["tvmaot"], f"Unsupported feature '{self.name}' for backend '{backend}'"
        return {f"{backend}.unpacked_api": self.enabled}


@register_feature("packed")
class Packed(FrameworkFeature, FrontendFeature, BackendFeature, SetupFeature):
    """Sub-8-bit and sparsity feature for TFLite Micro kernels."""

    def __init__(self, features=None, config=None):
        super().__init__("packed", features=features, config=config)

    def get_framework_config(self, framework):
        raise NotImplementedError

    def get_frontend_config(self, frontend):
        assert frontend in ["tflm"], f"Unsupported feature '{self.name} for frontend '{frontend}''"
        return {f"{frontend}.use_packed_weights": self.enabled}

    def get_backend_config(self, backend):
        raise NotImplementedError

    def get_required_cache_flags(self):
        return {"tflmc.exe": ["packed"]}


@register_feature("packing")
class Packing(FrontendFeature):
    """Sub-8-bit and sparse weight packing for TFLite Frontend."""

    def __init__(self, features=None, config=None):
        super().__init__("packing", features=features, config=config)

    def get_frontend_config(self, frontend):
        assert frontend in ["tflm"], f"Unsupported feature '{self.name} for frontend '{frontend}''"
        raise NotImplementedError
        return {f"{frontend}.pack_weights": self.enabled}


@register_feature("memplan")
class Memplan(FrameworkFeature):
    """Custom TVM memory planning feature by (@rafzi)"""

    def __init__(self, features=None, config=None):
        super().__init__("memplan", features=features, config=config)

    def get_framework_config(self, framework):
        assert framework in ["tvm"], f"Unsupported feature '{self.name}' for framework '{framework}'"
        raise NotImplementedError
        return {"tvm.memplan_enable": self.enabled}


@register_feature("usmp")
class Usmp(BackendFeature):
    """Unified Static Memory Planning algorithm integrated in TVM"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "algorithm": "greedy_by_conflicts",  # options: greedy_by_conflicts, greedy_by_size, hill_climb
    }

    def __init__(self, features=None, config=None):
        super().__init__("usmp", features=features, config=config)

    @property
    def algorithm(self):
        return str(self.config["algorithm"])

    def add_backend_config(self, backend, config):
        assert backend in ["tvmaot"], f"Unsupported feature '{self.name}' for backend '{backend}'"
        if f"{backend}.extra_pass_config" in config:
            tmp = config[f"{backend}.extra_pass_config"]
        elif "extra_pass_config" in config:
            tmp = config["extra_pass_config"]
        else:
            tmp = {}
        if isinstance(tmp, str):
            import ast

            tmp = ast.literal_eval(tmp)
        assert isinstance(tmp, dict)
        tmp["tir.usmp.enable"] = self.enabled
        tmp["tir.usmp.algorithm"] = self.algorithm
        config.update(
            {f"{backend}.extra_pass_config": tmp, f"{backend}.arena_size": 0}
        )  # In recent TVM versions USMP will have it's own arena.

    # -> enable this via backend


@register_feature("moiopt")
class MOIOPT(BackendFeature):
    """Memory-Optimizing, Inter-Operator Tiling - currently only supported with custom TVM"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "noftp": False,
        "onlyftp": False,
        "norecurse": False,
    }

    def __init__(self, features=None, config=None):
        super().__init__("moiopt", features=features, config=config)

    def add_backend_config(self, backend, config):
        assert backend in ["tvmaot", "tvmrt"], f"Unsupported feature '{self.name}' for backend '{backend}'"
        if f"{backend}.extra_pass_config" in config:
            tmp = config[f"{backend}.extra_pass_config"]
        elif "extra_pass_config" in config:
            tmp = config["extra_pass_config"]
        else:
            tmp = {}
        tmp["relay.moiopt.enable"] = self.enabled
        tmp["relay.moiopt.noftp"] = self.config["noftp"]
        tmp["relay.moiopt.onlyftp"] = self.config["onlyftp"]
        tmp["relay.moiopt.norecurse"] = self.config["norecurse"]
        config.update({f"{backend}.extra_pass_config": tmp})

    # -> enable this via backend


@register_feature("visualize")
class Visualize(FrontendFeature):
    """Visualize TFLite models."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
    }

    REQUIRED = ["tflite_visualize.exe"]

    def __init__(self, features=None, config=None):
        super().__init__("visualize", features=features, config=config)

    @property
    def tflite_visualize_exe(self):
        return self.config["tflite_visualize.exe"]

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name}' for frontend '{frontend}'"
        return filter_none(
            {
                f"{frontend}.visualize_enable": self.enabled,
                f"{frontend}.visualize_script": self.tflite_visualize_exe,
            }
        )

    def update_formats(self, frontend, input_formats, output_formats):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name}' for frontend '{frontend}'"
        if self.enabled:
            output_formats.append(ArtifactFormat.TEXT)


@register_feature("relayviz")
class Relayviz(FrontendFeature):
    """Visualize TVM relay models."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "plotter": "term",  # Alternative: dot
    }

    def __init__(self, features=None, config=None):
        super().__init__("relayviz", features=features, config=config)

    @property
    def plotter(self):
        return self.config.get("plotter", None)

    def get_frontend_config(self, frontend):
        assert frontend in ["relay"], f"Unsupported feature '{self.name}' for frontend '{frontend}'"
        return filter_none(
            {
                f"{frontend}.visualize_graph": self.enabled,
                f"{frontend}.relayviz_plotter": self.plotter,
            }
        )

    def update_formats(self, frontend, input_formats, output_formats):
        assert frontend in ["relay"], f"Unsupported feature '{self.name}' for frontend '{frontend}'"
        if self.enabled:
            output_formats.append(ArtifactFormat.TEXT)


@register_feature("autotuned")
class Autotuned(BackendFeature):
    """Use existing TVM autotuning logs in backend."""

    # TODO: FronendFeature to collect tuning logs or will we store them somewhere else?

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "results_file": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("autotuned", features=features, config=config)

    @property
    def results_file(self):
        return self.config["results_file"] if "results_file" in self.config else None

    def get_backend_config(self, backend):
        assert backend in SUPPORTED_TVM_BACKENDS
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

    def __init__(self, features=None, config=None):
        super().__init__("autotune", features=features, config=config)

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
        assert backend in SUPPORTED_TVM_BACKENDS
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


@register_feature("fallback")
class Fallback(FrameworkFeature, PlatformFeature):
    """(Unimplemented) TFLite Fallback for unsupported and custom operators in TVM."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "config_file": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("fallback", features=features, config=config)

    @property
    def config_file(self):
        return str(self.config["config_file"]) if "config_file" in self.config else None

    def get_framework_config(self, framework):
        assert framework in ["tvm"], f"Unsupported feature '{self.name}' for framework '{framework}'"
        raise NotImplementedError
        return filter_none(
            {
                f"{framework}.fallback_enable": self.enabled,
                f"{framework}.fallback_config_file": self.config_file,
            }
        )

    # -> hard to model..., preprocess for tflmc?


@register_feature("disable_legalize")
class DisableLegalize(BackendFeature, SetupFeature):
    """Enable transformation to reduces sizes of intermediate buffers by skipping legalization passes."""

    REQUIRED = ["tvm_extensions.wrapper"]

    def __init__(self, features=None, config=None):
        super().__init__("disable_legalize", features=features, config=config)

    @property
    def tvm_extensions_wrapper(self):
        return self.config["tvm_extensions.wrapper"]

    def add_backend_config(self, backend, config):
        assert backend in SUPPORTED_TVM_BACKENDS, f"Unsupported feature '{self.name}' for backend '{backend}'"
        if f"{backend}.tvmc_extra_args" in config:
            config[f"{backend}.tvmc_extra_args"].append("--disable-legalize")
        else:
            config[f"{backend}.tvmc_extra_args"] = ["--disable-legalize"]
        if f"{backend}.tvmc_custom_script" in config:
            assert config[f"{backend}.tvmc_custom_script"] is None or str(
                config[f"{backend}.tvmc_custom_script"]
            ) == str(
                self.tvm_extensions_wrapper
            ), f"{backend}.tvmc_custom_script is already set. Can't enable feature: {self.name}"
        config[f"{backend}.tvmc_custom_script"] = self.tvm_extensions_wrapper

    def get_required_cache_flags(self):
        ret = {}

        ret["tvm.pythonpath"] = ["patch"]
        return ret


@register_feature("demo")
class Demo(BackendFeature, SetupFeature):
    """Run demo application instead of benchmarking code."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "board": None,
        "print_stats": False,
        "print_interval_ms": 5000,
    }

    REQUIRED = []

    def __init__(self, features=None, config=None):
        super().__init__("demo", features=features, config=config)

    @property
    def board(self):
        return self.config["board"]

    @property
    def print_stats(self):
        return self.config["print_stats"]

    @property
    def print_interval_ms(self):
        return self.config["print_interval_ms"]

    def get_platform_defs(self, platform):
        assert platform in ["espidf"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        # TODO: espidf.demo_mode, disable wdt, runtime stats,
        return {}

    def get_platform_config(self, platform):
        assert platform in ["espidf"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        # TODO: espidf.demo_mode, disable wdt, runtime stats,
        return {}


@register_feature("cachesim")
class CacheSim(TargetFeature):
    """Collect information on cache misses etc. with spike target"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "ic_enable": False,
        "ic_config": "64:8:32",
        "dc_enable": False,
        "dc_config": "64:8:32",
        "l2_enable": False,
        "l2_config": "262144:8:32",  # TODO: find a meaningful value
        "log_misses": False,
        "detailed": False,
    }

    REQUIRED = []

    def __init__(self, features=None, config=None):
        super().__init__("cachesim", features=features, config=config)

    @property
    def ic_enable(self):
        return str2bool(self.config["ic_enable"])

    @property
    def ic_config(self):
        return self.config["ic_config"]

    @property
    def dc_enable(self):
        return str2bool(self.config["dc_enable"])

    @property
    def dc_config(self):
        return self.config["dc_config"]

    @property
    def l2_enable(self):
        return str2bool(self.config["l2_enable"])

    @property
    def l2_config(self):
        return self.config["l2_config"]

    @property
    def log_misses(self):
        return str2bool(self.config["log_misses"])

    @property
    def detailed(self):
        return str2bool(self.config["detailed"])

    def add_target_config(self, target, config):
        assert target in ["spike"], f"Unsupported feature '{self.name}' for target '{target}'"
        if self.enabled:
            spike_args = config.get(f"{target}.extra_args", [])
            if self.ic_enable:
                assert self.ic_config is not None and len(self.ic_config) > 0
                spike_args.append(f"--ic={self.ic_config}")
            if self.dc_enable:
                assert self.dc_config is not None and len(self.dc_config) > 0
                spike_args.append(f"--dc={self.dc_config}")
            if self.l2_enable:
                assert self.l2_config is not None and len(self.l2_config) > 0
                spike_args.append(f"--l2={self.l2_config}")
            if self.log_misses:
                spike_args.append("--log-cache-miss")
            config.update({f"{target}.extra_args": spike_args})

    def get_target_callback(self, target):
        assert target in ["spike"], f"Unsupported feature '{self.name}' for target '{target}'"
        if self.enabled:

            def cachesim_callback(stdout, metrics, artifacts):
                """Callback which parses the targets output and updates the generated metrics and artifacts."""
                expr = (
                    r"(D|I|L2)\$ ((?:Bytes (?:Read|Written))|(?:Read|Write) "
                    r"(?:Accesses|Misses)|(?:Writebacks)|(?:Miss Rate)):\s*(\d+\.?\d*%?)*"
                )
                matches = re.compile(expr).findall(stdout)
                prefixes = [
                    x for (x, y) in zip(["I", "D", "L2"], [self.ic_enable, self.dc_enable, self.l2_enable]) if y
                ]
                for groups in matches:
                    assert len(groups) == 3
                    prefix, label, value = groups
                    if not self.detailed:
                        if "Rate" not in label:
                            continue
                    value = int(value) if "%" not in value else float(value[:-1]) / 100
                    if prefix in prefixes:
                        metrics.add(f"{prefix}-Cache {label}", value)

            return cachesim_callback


@register_feature("log_instrs")
class LogInstructions(TargetFeature):
    """Enable logging of the executed instructions of a simulator-based target."""

    DEFAULTS = {**FeatureBase.DEFAULTS, "to_file": False}

    def __init__(self, features=None, config=None):
        super().__init__("log_instrs", features=features, config=config)

    @property
    def to_file(self):
        # TODO: implement get_bool_or_none?
        return bool(self.config["to_file"]) if self.config["to_file"] is not None else None

    def add_target_config(self, target, config):
        assert target in ["spike", "etiss_pulpino"]
        if not self.enabled:
            return
        if target == "spike":
            extra_args_new = config.get("extra_args", [])
            extra_args_new.append("-l")
            # if self.to_file:
            #     extra_args_new.append("--log=?")
            config.update({"extra_args": extra_args_new})
        elif target == "etiss_pulpino":
            plugins_new = config.get("plugins", [])
            plugins_new.append("PrintInstruction")
            config.update({f"{target}.plugins": plugins_new})

    def get_target_callback(self, target):
        assert target in ["spike", "etiss_pulpino"], f"Unsupported feature '{self.name}' for target '{target}'"
        if self.enabled:

            def log_instrs_callback(stdout, metrics, artifacts):
                """Callback which parses the targets output and updates the generated metrics and artifacts."""
                if self.to_file:
                    # TODO: update stdout and remove log_instrs lines
                    instrs = []
                    for line in stdout.split("\n"):
                        if target == "etiss_pulpino":
                            expr = re.compile(r"0x[a-fA-F0-9]+: .* \[.*\]")
                        elif target == "spike":
                            expr = re.compile(r"core\s+\d+: 0x[a-fA-F0-9]+ \(0x[a-fA-F0-9]+\) .*")
                        match = expr.match(line)
                        if match is not None:
                            instrs.append(line)
                    instrs_artifact = Artifact(
                        f"{target}_instrs.log", content="\n".join(instrs), fmt=ArtifactFormat.TEXT
                    )
                    artifacts.append(instrs_artifact)

            return log_instrs_callback


@register_feature("microtvm_etissvp")
class MicrotvmEtissVp(PlatformFeature):
    """Use ETISS VP for MicroTVM deployment in TVM."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "verbose": False,
        "debug": False,
        "transport": True,
    }

    REQUIRED = ["microtvm_etissvp.template", "etiss.install_dir", "etissvp.script", "riscv_gcc.install_dir"]

    def __init__(self, features=None, config=None):
        super().__init__("microtvm_etissvp", features=features, config=config)

    @property
    def microtvm_etissvp_template(self):
        return self.config["microtvm_etissvp.template"]

    @property
    def etiss_install_dir(self):
        return self.config["etiss.install_dir"]

    @property
    def etissvp_script(self):
        return self.config["etissvp.script"]

    @property
    def riscv_gcc_install_dir(self):
        return self.config["riscv_gcc.install_dir"]

    @property
    def verbose(self):
        return str2bool(self.config["verbose"])

    @property
    def debug(self):
        return str2bool(self.config["debug"])

    @property
    def transport(self):
        return str2bool(self.config["transport"])

    def get_platform_config(self, platform):
        assert platform == "microtvm", f"Unsupported feature '{self.name}' for platform '{platform}'"
        etissvp_ini = Path(self.microtvm_etissvp_template) / "scripts" / "memsegs.ini"

        project_options = {
            "project_type": "host_driven",
            "verbose": str(self.verbose).lower(),
            "debug": str(self.debug).lower(),
            "transport": str(self.transport).lower(),
            "etiss_path": str(self.etiss_install_dir),
            "riscv_path": str(self.riscv_gcc_install_dir),
            "etissvp_script": str(self.etissvp_script),
            "etissvp_script_args": f"plic clint uart v -i{etissvp_ini}",  # TODO: remove v
        }

        return filter_none(
            {
                f"{platform}.project_template": self.microtvm_etissvp_template,
                f"{platform}.project_options": project_options,
            }
        )


@register_feature("arm_mvei")
class ArmMvei(SetupFeature, TargetFeature, PlatformFeature):
    """Enable MVEI extension for supported ARM targets"""

    def __init__(self, features=None, config=None):
        super().__init__("arm_mvei", features=features, config=config)

    def get_target_config(self, target):
        assert target in ["corstone300"]
        return {
            f"{target}.enable_mvei": True,  # TODO: remove if not required (only enforce m33/m55)
        }

    def get_required_cache_flags(self):
        return {
            "cmsisnn.lib": ["mvei"],
            "tflmc.exe": ["mvei"],
        }

    def get_platform_defs(self, platform):
        return {"ARM_MVEI": self.enabled}


@register_feature("arm_dsp")
class ArmDsp(SetupFeature, TargetFeature, PlatformFeature):
    """Enable DSP extension for supported ARM targets"""

    def __init__(self, features=None, config=None):
        super().__init__("arm_dsp", features=features, config=config)

    def get_target_config(self, target):
        assert target in ["corstone300"]
        return {
            f"{target}.enable_dsp": True,  # TODO: remove if not required (only enforce m33/m55)
        }

    def get_required_cache_flags(self):
        # These will be merged automatically with existing ones
        return {
            "cmsisnn.lib": ["dsp"],
            "tflmc.exe": ["dsp"],
        }

    def get_platform_defs(self, platform):
        return {"ARM_DSP": self.enabled}


@register_feature("target_optimized")
class TargetOptimized(RunFeature):
    """Overwrite backend options according to chosen target."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
    }

    def __init__(self, features=None, config=None):
        super().__init__("target_optimized", features=features, config=config)

    def get_run_config(self):
        return {"run.target_to_backend": self.enabled}
