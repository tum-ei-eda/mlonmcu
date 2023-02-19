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
import pandas as pd
from typing import Union

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
    "tvmaotplus",
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
            "tvmaotplus",
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
        value = self.config["allow_missing"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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
        "elen": 32,
        # use target-side settings by default
        "spec": None,
        "embedded": None,
    }

    REQUIRED = []

    def __init__(self, features=None, config=None):
        super().__init__("vext", features=features, config=config)

    @property
    def vlen(self):
        return int(self.config["vlen"])

    @property
    def elen(self):
        return int(self.config["elen"])

    @property
    def spec(self):
        return self.config["spec"]

    @property
    def embedded(self):
        return self.config["embedded"]

    def get_target_config(self, target):
        # TODO: enforce llvm toolchain using add_compile_config and CompileFeature?
        assert is_power_of_two(self.vlen)
        return filter_none(
            {
                f"{target}.enable_vext": True,
                f"{target}.vlen": self.vlen,
                f"{target}.elen": self.elen,
                f"{target}.vext_spec": self.spec,
                f"{target}.embedded_vext": self.embedded,
            }
        )

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
            "riscv_gcc.name": ["vext"],
        }


@register_feature("pext")
class Pext(SetupFeature, TargetFeature, PlatformFeature):
    """Enable packed SIMD extension for supported RISC-V targets"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        # use target-side settings by default
        "spec": None,
    }

    REQUIRED = []

    def __init__(self, features=None, config=None):
        super().__init__("pext", features=features, config=config)

    @property
    def spec(self):
        return self.config["spec"]

    def get_target_config(self, target):
        return filter_none(
            {
                f"{target}.enable_pext": True,  # Handle via arch characters in the future
                f"{target}.pext_spec": self.spec,
            }
        )

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {"RISCV_PEXT": self.enabled}

    def get_required_cache_flags(self):
        # These will be merged automatically with existing ones
        return {
            "muriscvnn.lib": ["pext"],
            "tflmc.exe": ["pext"],
            "riscv_gcc.install_dir": ["pext"],
            "riscv_gcc.name": ["pext"],
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
        value = self.config["attach"]
        return str2bool(value, allow_none=True) if not isinstance(value, (bool, int)) else value

    @property
    def port(self):
        return int(self.config["port"]) if self.config["port"] is not None else None

    def get_target_config(self, target):
        assert target in ["host_x86", "etiss_pulpino", "ovpsim"]
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

    DEFAULTS = {**FeatureBase.DEFAULTS, "to_file": True}  # ETISS can only trace to file

    @property
    def to_file(self):
        value = self.config["to_file"]
        return str2bool(value, allow_none=True) if not isinstance(value, (bool, int)) else value

    def __init__(self, features=None, config=None):
        super().__init__("trace", features=features, config=config)

    def add_target_config(self, target, config):
        assert target in ["etiss_pulpino", "ovpsim"]
        if target == "etiss_pulpino":
            config.update({"etiss_pulpino.trace_memory": self.enabled})
        elif target == "ovpsim":
            extra_args_new = config.get("extra_args", [])
            extra_args_new.append("--trace --tracemem SAX")
            # if self.to_file:
            #    extra_args_new.append("--tracefile")
            config.update({f"{target}.extra_args": extra_args_new})


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
        if self.algorithm in ["greedy_by_size", "greedy_by_conflicts", "hill_climb"]:
            tmp["tir.usmp.algorithm"] = self.algorithm
        else:
            tmp["tir.usmp.custom_algorithm"] = self.algorithm
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
        "maxpartitions": 0,
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
        tmp["relay.moiopt.maxpartitions"] = self.config["maxpartitions"]
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
class Autotune(PlatformFeature, RunFeature):
    """Use the TVM autotuner inside the backend to generate tuning logs."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "results_file": None,
        "append": None,
        "tuner": None,
        "trials": None,
        "early_stopping": None,
        "num_workers": None,
        "max_parallel": None,
        "use_rpc": None,
        "timeout": None,
        "mode": None,
        "visualize": None,
        "tasks": None,
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

    @property
    def mode(self):
        return self.config["mode"]

    @property
    def visualize(self):
        return self.config["visualize"]

    @property
    def tasks(self):
        return self.config["tasks"]

    def get_platform_config(self, platform):
        assert platform in ["tvm", "microtvm"]
        # TODO: figure out a default path automatically
        return filter_none(
            {
                f"{platform}.autotuning_enable": self.enabled,
                f"{platform}.autotuning_results_file": self.results_file,
                f"{platform}.autotuning_append": self.append,
                f"{platform}.autotuning_tuner": self.tuner,
                f"{platform}.autotuning_trials": self.trials,
                f"{platform}.autotuning_early_stopping": self.early_stopping,
                f"{platform}.autotuning_num_workers": self.num_workers,
                f"{platform}.autotuning_max_parallel": self.max_parallel,
                f"{platform}.autotuning_timeout": self.timeout,
                f"{platform}.autotuning_mode": self.mode,
                f"{platform}.autotuning_visualize": self.visualize,
                f"{platform}.autotuning_tasks": self.tasks,
            }
        )

    def get_run_config(self):
        return {"run.tune_enabled": self.enabled}


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
class Demo(PlatformFeature):
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
        value = self.config["ic_enable"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def ic_config(self):
        return self.config["ic_config"]

    @property
    def dc_enable(self):
        value = self.config["dc_enable"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def dc_config(self):
        return self.config["dc_config"]

    @property
    def l2_enable(self):
        value = self.config["l2_enable"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def l2_config(self):
        return self.config["l2_config"]

    @property
    def log_misses(self):
        value = self.config["log_misses"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def detailed(self):
        value = self.config["detailed"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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

    def get_target_callbacks(self, target):
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
                        for m in metrics:
                            m.add(f"{prefix}-Cache {label}", value)
                return stdout

            return None, cachesim_callback


@register_feature("log_instrs")
class LogInstructions(TargetFeature):
    """Enable logging of the executed instructions of a simulator-based target."""

    DEFAULTS = {**FeatureBase.DEFAULTS, "to_file": False}

    def __init__(self, features=None, config=None):
        super().__init__("log_instrs", features=features, config=config)

    @property
    def to_file(self):
        value = self.config["to_file"]
        return str2bool(value, allow_none=True) if not isinstance(value, (bool, int)) else value

    def add_target_config(self, target, config):
        assert target in ["spike", "etiss_pulpino", "ovpsim", "gvsoc_pulp"]
        if not self.enabled:
            return
        if target == "spike":
            extra_args_new = config.get("extra_args", [])
            extra_args_new.append("-l")
            # if self.to_file:
            #     extra_args_new.append("--log=?")
            config.update({f"{target}.extra_args": extra_args_new})
        elif target == "etiss_pulpino":
            plugins_new = config.get("plugins", [])
            plugins_new.append("PrintInstruction")
            config.update({f"{target}.plugins": plugins_new})
        elif target == "ovpsim":
            extra_args_new = config.get("extra_args", [])
            extra_args_new.append("--trace")
            # if self.to_file:
            #    extra_args_new.append("--tracefile")
            config.update({f"{target}.extra_args": extra_args_new})
        elif target == "gvsoc_pulp":
            extra_args_new = config.get("extra_args", [])
            if self.to_file:
                extra_args_new.append(f"--trace=insn:{target}_instrs.log")
                """
                TODO: The above code will generate a instruction log.
                But it will not be recorded by Artifact (which should be done in get_target_callbacks).
                The code to let it be recorded by Artifact is currently not added because of the following reasons:
                1. This feature is rarely used.
                2. GVSOC can directly write the instruction log into a file which should be much faster than
                read/write from the output. This makes GVSOC special and difficult to be adapted in the current
                code.
                3. This difficulty reflected in that the methods add_target_config and get_target_callbacks should
                have a consensus about where in the system file system the instruction log is written to and can be
                read from. This is not feasible in current code and the realization requires a workaround.
                """
            else:
                extra_args_new.append("--trace=insn")
            config.update({f"{target}.extra_args": extra_args_new})

    def get_target_callbacks(self, target):
        assert target in [
            "spike",
            "etiss_pulpino",
            "ovpsim",
            "gvsoc_pulp",
        ], f"Unsupported feature '{self.name}' for target '{target}'"
        if self.enabled:
            if not target == "gvsoc_pulp":

                def log_instrs_callback(stdout, metrics, artifacts):
                    """Callback which parses the targets output and updates the generated metrics and artifacts."""
                    new_lines = []
                    if self.to_file:
                        # TODO: update stdout and remove log_instrs lines
                        instrs = []
                        for line in stdout.split("\n"):
                            if target == "etiss_pulpino":
                                expr = re.compile(r"0x[a-fA-F0-9]+: .* \[.*\]")
                            elif target == "spike":
                                expr = re.compile(r"core\s+\d+: 0x[a-fA-F0-9]+ \(0x[a-fA-F0-9]+\) .*")
                            elif target == "ovpsim":
                                expr = re.compile(
                                    r"Info 'riscvOVPsim\/cpu',\s0x[0-9abcdef]+\(.*\):\s[0-9abcdef]+\s+\w+\s+.*"
                                )
                            match = expr.match(line)
                            if match is not None:
                                instrs.append(line)
                            else:
                                new_lines.append(line)
                        instrs_artifact = Artifact(
                            f"{target}_instrs.log",
                            content="\n".join(instrs),
                            fmt=ArtifactFormat.TEXT,
                            flags=(self.name, target),
                        )
                        artifacts.append(instrs_artifact)
                        return "\n".join(new_lines)
                    else:
                        return stdout

                return None, log_instrs_callback
        return None, None


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


# Needs: vext
# RISC-V only
# Warning: Auto-vectorization is turned on by default quite low optimization levels
# Therfore this feature is mainly for debugging the auto-vectorization procedure
@register_feature("auto_vectorize")
class AutoVectorize(PlatformFeature):
    """Enable auto_vectorization for supported MLIF platform targets."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "verbose": False,
        "loop": True,
        "slp": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("auto_vectorize", features=features, config=config)

    @property
    def verbose(self):
        value = self.config["verbose"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def loop(self):
        value = self.config["loop"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def slp(self):
        value = self.config["slp"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    def get_platform_defs(self, platform):
        return {
            "RISCV_AUTO_VECTORIZE": self.enabled,
            "RISCV_AUTO_VECTORIZE_VERBOSE": self.verbose,
            "RISCV_AUTO_VECTORIZE_LOOP": self.loop and self.enabled,
            "RISCV_AUTO_VECTORIZE_SLP": self.slp and self.enabled,
        }


@register_feature("benchmark")
class Benchmark(PlatformFeature, TargetFeature):
    """Profile code using supported platforms."""

    # TODO: would make sense to move end_to_end here as well!

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "num_runs": 1,
        "num_repeat": 1,
        "total": False,
        "aggregate": "avg",  # Allowed: avg, max, min, none, all
    }

    REQUIRED = []

    def __init__(self, features=None, config=None):
        super().__init__("benchmark", features=features, config=config)

    @property
    def num_runs(self):
        return int(self.config["num_runs"])

    @property
    def num_repeat(self):
        return int(self.config["num_repeat"])

    @property
    def total(self):
        value = self.config["total"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def aggregate(self):
        value = self.config["aggregate"]
        assert value in ["avg", "all", "max", "min", "none"]
        return value

    def get_platform_config(self, platform):
        supported = ["mlif", "tvm"]  # TODO: support microtvm and espidf
        assert platform in supported, f"Unsupported feature '{self.name}' for platform '{platform}'"

        if platform in ["tvm", "microtvm"]:
            return {
                f"{platform}.number": self.num_runs,
                f"{platform}.repeat": self.num_repeat,
                f"{platform}.aggregate": self.aggregate,
                f"{platform}.total_time": self.total,
            }
        else:
            return {}

    def get_target_config(self, target):
        return {
            f"{target}.repeat": self.num_repeat,
        }

    def get_platform_defs(self, platform):
        supported = ["mlif", "espidf", "tvm", "microtvm", "zephyr"]  # TODO: support microtvm and espidf
        assert platform in supported, f"Unsupported feature '{self.name}' for platform '{platform}'"

        if platform == "mlif":
            return {"NUM_RUNS": self.num_runs}
        elif platform == "espidf":
            return {"MLONMCU_NUM_RUNS": self.num_runs}
        else:
            return {}

    def get_target_callbacks(self, target):
        if self.enabled:

            def benchmark_callback(stdout, metrics, artifacts):
                if len(metrics) <= 1:
                    return
                metrics_ = metrics[1:]  # drop first run (warmup)

                # TODO: this currently processes all numeric metrics, should probably ignore stuff like MIPS etc.
                data_ = [
                    {
                        key: (float(value) / self.num_runs) if self.num_runs > 1 else value
                        for key, value in m.data.items()
                        if "cycle" in key.lower() or "time" in key.lower()
                    }
                    for m in metrics_
                ]

                df = pd.DataFrame(data_)

                if self.aggregate == "all":
                    aggs = ["mean", "min", "max"]
                elif self.aggregate in ["avg", "mean"]:
                    aggs = ["mean"]
                elif self.aggregate == "min":
                    aggs = ["min"]
                elif self.aggregate == "max":
                    aggs = ["max"]
                elif self.aggregate == "none":
                    aggs = []

                if len(aggs) == 0:
                    data = {}
                else:
                    df_ = df.agg(aggs)

                    # rename columns
                    index_mapping = {
                        "mean": "Average",
                        "min": "Min",
                        "max": "Max",
                    }

                    df_ = df_.rename(index=index_mapping)

                    data = df_.to_dict()

                    data = {f"{prefix} {key}": value for key, temp in data.items() for prefix, value in temp.items()}
                if self.total:
                    data.update(
                        {
                            f"Total {key}": (value * self.num_runs) if self.num_runs > 1 else value
                            for key, value in data_[-1].items()
                            if "cycle" in key.lower() or "time" in key.lower()
                        }
                    )
                metrics_ = metrics_[-1]
                metrics_.data.update(data)

                for key in data_[-1].keys():
                    if key in metrics_.order:
                        if self.total:
                            metrics_.order.append(f"Total {key}")
                        metrics_.order.remove(key)
                for key in data.keys():
                    if key not in metrics_.order:
                        metrics_.order.append(key)
                metrics.clear()
                metrics.append(metrics_)
                return stdout

            benchmark_callback.priority = 0

            return None, benchmark_callback


@register_feature("tvm_rpc")
class TvmRpc(PlatformFeature):
    """Run TVM models on a RPC device."""

    DEFAULTS = {**FeatureBase.DEFAULTS, "hostname": None, "port": None, "key": None}  # tracker

    def __init__(self, features=None, config=None):
        super().__init__("tvm_rpc", features=features, config=config)

    @property
    def use_rpc(self):
        return self.config["use_rpc"]

    @property
    def hostname(self):
        return self.config["hostname"]

    @property
    def port(self):
        return self.config["port"]

    @property
    def key(self):
        return self.config["key"]

    def get_platform_config(self, platform):
        assert platform in ["tvm", "microtvm"]
        return filter_none(
            {
                f"{platform}.use_rpc": self.enabled,
                f"{platform}.rpc_hostname": self.hostname,
                f"{platform}.rpc_port": self.port,
                f"{platform}.rpc_key": self.key,
            }
        )


@register_feature("tvm_profile")
class TvmProfile(PlatformFeature):
    """Profile code using TVM Platform."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
    }

    REQUIRED = []

    def __init__(self, features=None, config=None):
        super().__init__("tvm_profile", features=features, config=config)

    def get_platform_config(self, platform):
        supported = ["tvm", "microtvm"]
        assert platform in supported, f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {
            f"{platform}.profile": self.enabled,
        }


@register_feature("xpulp")
class Xpulp(TargetFeature, PlatformFeature, SetupFeature):
    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "xpulp_version": 2,
        "nopostmod": False,
        "noindregreg": False,
        "novect": False,
        "nohwloop": False,
        "hwloopmin": 2,
        "hwloopalign": False,
        "nomac": False,
        "nopartmac": False,
        "nominmax": False,
        "noabs": False,
        "nobitop": False,
        "nosext": False,
        "noclip": False,
        "noaddsubnormround": False,
        "noshufflepack": False,
        "nomulmacnormround": False,
        "noshufflepack": False,
    }

    REQUIRED = ["pulp_gcc.install_dir", "pulp_gcc.name"]

    def __init__(self, features=None, config=None):
        super().__init__("xpulp", features=features, config=config)

    # Except the "enabled" in FeatureBase.DEFAULTS and "xpulp_version"
    # every key in DEFAULTS should in principle have a getter function with the same name as the key
    # These getter functions will be stored in getter_functions array.
    # Default getter function for the keys whose corresponding value has bool type:
    # def getter_bool(self, key_name):
    #    return self.generalized_str2bool(self.config[key_name])
    # Default getter function for the keys whose corresponding value has int type:
    # def getter_int(self, key_name):
    #    return int(self.config["<key_name>"])
    # No default, i.e. customized getter functions are defined separately

    # Default getter functions:
    @staticmethod
    def generalized_str2bool(input: Union[str, bool, int]) -> bool:
        return str2bool(input) if isinstance(input, str) else input

    def getter_bool(self, key_name: Union[str, bool, int]) -> bool:
        return self.generalized_str2bool(self.config[key_name])

    def getter_int(self, key_name: Union[str, bool, int]) -> int:
        return int(self.config[key_name])

    # No default, i.e. customized getter functions are defined in the following (now empty)

    # custom_config_getter contains customized @property function which do not follow the pattern above.
    getter_functions = {
        "nopostmod": getter_bool,
        "noindregreg": getter_bool,
        "novect": getter_bool,
        "nohwloop": getter_bool,
        "hwloopmin": getter_int,
        "hwloopalign": getter_bool,
        "nomac": getter_bool,
        "nopartmac": getter_bool,
        "nominmax": getter_bool,
        "noabs": getter_bool,
        "nobitop": getter_bool,
        "nosext": getter_bool,
        "noclip": getter_bool,
        "noaddsubnormround": getter_bool,
        "noshufflepack": getter_bool,
        "nomulmacnormround": getter_bool,
        "noshufflepack": getter_bool,
    }

    @property
    def xpulp_version(self):
        value = self.config["xpulp_version"]
        value = int(value) if not isinstance(value, int) else value  # convert to int
        assert value in [None, 2, 3], f"xpulp_version must be None, 2 or 3, but get {value}"
        return value

    def get_platform_defs(self, platform):
        # The following create EXTRA_FLAGS (type is str) for gcc
        # example
        # {"nopostmod": True, "novect": True, ...} ==> EXTRA_FLAGS = "-mnopostmod -mnovect ..."
        EXTRA_FLAGS = ""
        for key in self.getter_functions:
            if isinstance(self.getter_functions[key](self, key), bool):
                if self.getter_functions[key](self, key):
                    EXTRA_FLAGS += f" -m{key}"
                continue
            if isinstance(self.getter_functions[key](self, key), int):
                EXTRA_FLAGS += f" -m{key}={self.getter_functions[key](self, key)}"
                continue
        EXTRA_FLAGS = "'" + EXTRA_FLAGS.strip() + "'"
        return {
            # EXTRA_CMAKE_C_FLAGS will be directly append to CMAKE_C_FLAGS in mlonmcu_sw/mlif/tootchains/Pulp.cmake
            "EXTRA_CMAKE_C_FLAGS": EXTRA_FLAGS,
            # EXTRA_CMAKE_CXX_FLAGS will be directly append to CMAKE_CXX_FLAGS in mlonmcu_sw/mlif/tootchains/Pulp.cmake
            "EXTRA_CMAKE_CXX_FLAGS": EXTRA_FLAGS,
        }

    def add_platform_defs(self, platform, defs):
        addition_defs = self.get_platform_defs(platform)
        self.merge_dicts(defs, addition_defs)

    @staticmethod
    def merge_dicts(dict1, dict2):
        """
        This function tries to merge dict1 and dict2 into dict1
        :param dict1: A dictionary
        :param dict2: A dictionary to be added
        :return: Void
        Example 1:
        dict1 = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        dict2 = {"f": 3, "b": "world", "c": [4, 5, 6]}
        merge_dicts(dict1, dict2)
        print(dict1)
        ==>
        {"a": 1, "b": "hello world", "c": [1, 2, 3, 4, 5, 6], "f":3}
        Note: Here "hello" and "world" are merged as two string join.
        Here [1,2,3] and [4,5,6] are merged as list addition
        Example 2:
        dict1 = {"a": 1}
        dict2 = {"a": 3}
        merge_dicts(dict1, dict2)
        ==>
        RuntimeError: The method to merge a: 1 and a: 3 is not defined
        """
        for key in dict2.keys():
            if key in dict1.keys():
                dict1_value = dict1[key]
                dict2_value = dict2[key]
                if isinstance(dict1_value, (str, list)) and type(dict1_value) == type(dict2_value):
                    if isinstance(dict1_value, str):
                        dict1[key] = dict1_value + " " + dict2_value
                    else:
                        dict1[key] = dict1_value + dict2_value
                else:
                    raise RuntimeError(
                        f"The method to merge {key}: {dict1_value} and {key}: {dict2_value} is not defined"
                    )
            else:
                dict1[key] = dict2[key]

    def get_target_config(self, target):
        return filter_none({f"{target}.xpulp_version": self.xpulp_version})


@register_feature("split_layers")
class SplitLayers(FrontendFeature):
    """Split TFLite models into subruns."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
    }

    REQUIRED = ["tflite_pack.exe"]

    def __init__(self, features=None, config=None):
        super().__init__("split_layers", features=features, config=config)

    @property
    def tflite_pack_exe(self):
        return self.config["tflite_pack.exe"]

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name}' for frontend '{frontend}'"
        return filter_none(
            {
                f"{frontend}.split_layers": self.enabled,
                f"{frontend}.pack_script": self.tflite_pack_exe,
            }
        )
