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
from pathlib import Path

from mlonmcu.utils import is_power_of_two, filter_none
from mlonmcu.config import str2bool, str2list
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


REGISTERED_FEATURES = {}
FEATURE_DEPS = {}


def register_feature(name, depends=None):
    """Decorator for adding a feature to the global registry."""
    if depends is None:
        depends = []

    def real_decorator(obj):
        REGISTERED_FEATURES[name] = obj
        FEATURE_DEPS[name] = depends

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


def get_available_features(feature_type=None, feature_name=None, deps=False):
    """Utility for looking up features."""
    names = get_available_feature_names(feature_type=feature_type)
    names = [name for name in names if feature_name is None or name == feature_name]
    ret = {}
    if deps:
        for name in names:
            names = list(set(names + FEATURE_DEPS[name]))
    for name in names:
        ret[name] = REGISTERED_FEATURES[name]
    return ret


def get_matching_features(features, feature_type):
    return [feature for feature in features if feature_type in feature.types()]


# @register_feature("debug_arena", depends=["debug"])
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


# @register_feature("validate", depends=["debug"])
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
        return str2bool(value)

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

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "use_vext": "AUTO",
        "use_pext": "AUTO",
    }

    REQUIRED = {"muriscvnn.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__("muriscvnn", features=features, config=config)

    @property
    def muriscvnn_dir(self):
        return str(self.config["muriscvnn.src_dir"])

    @property
    def use_vext(self):
        value = self.config["use_vext"]
        if value == "AUTO" or value is None:
            return value
        value = str2bool(value)
        return "ON" if value else "OFF"

    @property
    def use_pext(self):
        value = self.config["use_pext"]
        if value == "AUTO" or value is None:
            return value
        value = str2bool(value)
        return "ON" if value else "OFF"

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
            "MURISCVNN_VEXT": self.use_vext,
            "MURISCVNN_PEXT": self.use_pext,
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

    REQUIRED = {"cmsisnn.dir", "cmsis.dir"}

    def __init__(self, features=None, config=None):
        super().__init__("cmsisnn", features=features, config=config)

    @property
    def cmsisnn_dir(self):
        return str(self.config["cmsisnn.dir"])

    @property
    def cmsis_dir(self):
        return str(self.config["cmsis.dir"])

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
            "CMSIS_DIR": self.cmsis_dir,
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
        "mattr": None,  # for +nodsp, +nomve
        "debug_last_error": False,
    }

    REQUIRED = {"cmsisnn.dir", "cmsis.dir"}

    def __init__(self, features=None, config=None):
        super().__init__("cmsisnnbyoc", features=features, config=config)

    @property
    def cmsisnn_dir(self):
        return str(self.config["cmsisnn.dir"])

    @property
    def cmsis_dir(self):
        return str(self.config["cmsis.dir"])

    @property
    def mcpu(self):
        return self.config["mcpu"]

    @property
    def mattr(self):
        return self.config["mattr"]

    @property
    def debug_last_error(self):
        return str2bool(self.config["debug_last_error"])

    def add_backend_config(self, backend, config):
        assert backend in SUPPORTED_TVM_BACKENDS, f"Unsupported feature '{self.name}' for backend '{backend}'"
        extras = config.get(f"{backend}.extra_targets", [])
        if extras is None:
            extras = []
        if "cmsis-nn" not in extras:
            if isinstance(extras, str):
                extras = str2list(extras)
            extras.append("cmsis-nn")
        config[f"{backend}.extra_targets"] = extras
        # Ideally cmsisnnbyoc would have a mvei/dsp feature which could be used to set this automatically
        extra_target_details = config.get(f"{backend}.extra_target_details", {})
        if extra_target_details is None:
            extra_target_details = {}
        cmsisnn_target_details = extra_target_details.get("cmsis-nn", {})
        if self.mcpu:
            cmsisnn_target_details["mcpu"] = self.mcpu
        if self.mattr:
            cmsisnn_target_details["mattr"] = self.mattr
        if self.debug_last_error is not None:
            cmsisnn_target_details["debug_last_error"] = self.debug_last_error
        extra_target_details["cmsis-nn"] = cmsisnn_target_details

        config[f"{backend}.extra_target_details"] = extra_target_details

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {
            "CMSISNN": self.enabled,
            "CMSIS_DIR": self.cmsis_dir,
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
        "mattr": None,  # for +nodsp, +nomve
        "debug_last_error": False,
        "use_vext": "AUTO",
        "use_pext": "AUTO",
    }

    REQUIRED = {"muriscvnn.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__("muriscvnnbyoc", features=features, config=config)

    @property
    def muriscvnn_dir(self):
        return str(self.config["muriscvnn.src_dir"])

    @property
    def mcpu(self):
        return self.config["mcpu"]

    @property
    def mattr(self):
        return self.config["mattr"]

    @property
    def debug_last_error(self):
        return str2bool(self.config["debug_last_error"])

    @property
    def use_vext(self):
        value = self.config["use_vext"]
        if value == "AUTO" or value is None:
            return value
        value = str2bool(value)
        return "ON" if value else "OFF"

    @property
    def use_pext(self):
        value = self.config["use_pext"]
        if value == "AUTO" or value is None:
            return value
        value = str2bool(value)
        return "ON" if value else "OFF"

    def add_backend_config(self, backend, config):
        assert backend in SUPPORTED_TVM_BACKENDS, f"Unsupported feature '{self.name}' for backend '{backend}'"
        extras = config.get(f"{backend}.extra_targets", [])
        if extras is None:
            extras = []
        if "cmsis-nn" not in extras:
            if isinstance(extras, str):
                extras = str2list(extras)
            extras.append("cmsis-nn")
        config[f"{backend}.extra_targets"] = extras
        # Ideally cmsisnnbyoc would have a mvei/dsp feature which could be used to set this automatically
        extra_target_details = config.get(f"{backend}.extra_target_details", {})
        if extra_target_details is None:
            extra_target_details = {}
        cmsisnn_target_details = extra_target_details.get("cmsis-nn", {})
        if self.mcpu:
            cmsisnn_target_details["mcpu"] = self.mcpu
        if self.mattr:
            cmsisnn_target_details["mattr"] = self.mattr
        if self.debug_last_error is not None:
            cmsisnn_target_details["debug_last_error"] = self.debug_last_error
        extra_target_details["cmsis-nn"] = cmsisnn_target_details

        config[f"{backend}.extra_target_details"] = extra_target_details

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {
            "MURISCVNN": self.enabled,
            "MURISCVNN_DIR": self.muriscvnn_dir,
            "MURISCVNN_VEXT": self.use_vext,
            "MURISCVNN_PEXT": self.use_pext,
        }

    def get_required_cache_flags(self):
        ret = {}
        ret["tvm.build_dir"] = ["cmsisnn"]
        return ret


VEXT_MIN_ALLOWED_VLEN = 64


# @before_feature("muriscvnn")  # TODO: implement something like this
@register_feature("vext")
class Vext(SetupFeature, TargetFeature, PlatformFeature):
    """Enable vector extension for supported RISC-V targets"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        # None -> use target setting
        "vlen": None,  # 64 does not work with every toolchain
        "elen": None,  # some toolchains may generate auto-vectorized programs with elen 64
        # use target-side settings by default
        "spec": None,
        "embedded": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("vext", features=features, config=config)

    @property
    def vlen(self):
        value = self.config["vlen"]
        return None if value is None else int(value)

    @property
    def elen(self):
        value = self.config["elen"]
        return None if value is None else int(value)

    @property
    def spec(self):
        return self.config["spec"]

    @property
    def embedded(self):
        return self.config["embedded"]

    def get_target_config(self, target):
        # TODO: enforce llvm toolchain using add_compile_config and CompileFeature?
        assert self.vlen is None or is_power_of_two(self.vlen)
        assert self.vlen is None or self.vlen >= VEXT_MIN_ALLOWED_VLEN
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
        return {
            "RISCV_VEXT": self.enabled,
            "RISCV_VLEN": self.vlen,
        }

    def get_required_cache_flags(self):
        return {
            "muriscvnn.lib": ["vext"],
            "tflmc.exe": ["vext"],
            # "riscv_gcc.install_dir": ["vext"],
            # "riscv_gcc.name": ["vext"],
        }


@register_feature("pext")
class Pext(SetupFeature, TargetFeature, PlatformFeature):
    """Enable packed SIMD extension for supported RISC-V targets"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        # use target-side settings by default
        "spec": None,
    }

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


@register_feature("bext")
class Bext(SetupFeature, TargetFeature, PlatformFeature):
    """Enable bitmanipulation extension for supported RISC-V targets"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        # use target-side settings by default
        "spec": None,
        "zba": True,
        "zbb": True,
        "zbc": True,
        "zbs": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("bext", features=features, config=config)

    @property
    def spec(self):
        return self.config["spec"]

    @property
    def zba(self):
        value = self.config["zba"]
        return str2bool(value)

    @property
    def zbb(self):
        value = self.config["zbb"]
        return str2bool(value)

    @property
    def zbc(self):
        value = self.config["zbc"]
        return str2bool(value)

    @property
    def zbs(self):
        value = self.config["zbs"]
        return str2bool(value)

    def get_target_config(self, target):
        return filter_none(
            {
                f"{target}.enable_bext": True,  # Handle via arch characters in the future
                f"{target}.bext_spec": self.spec,
                f"{target}.bext_zba": self.zba,
                f"{target}.bext_zbb": self.zbb,
                f"{target}.bext_zbc": self.zbc,
                f"{target}.bext_zbs": self.zbs,
            }
        )

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {"RISCV_PEXT": self.enabled}

    def get_required_cache_flags(self):
        # These will be merged automatically with existing ones
        return {
            # "riscv_gcc.install_dir": ["bext"],
            # "riscv_gcc.name": ["bext"],
            "riscv_gcc.install_dir": [],
            "riscv_gcc.name": [],
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
        return str2bool(value, allow_none=True)

    @property
    def port(self):
        return int(self.config["port"]) if self.config["port"] is not None else None

    def get_target_config(self, target):
        assert target in ["host_x86", "etiss_pulpino", "etiss", "ovpsim", "corev_ovpsim"]
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
        assert target in ["etiss_pulpino", "etiss"]
        return {f"{target}.debug_etiss": self.enabled}


@register_feature("trace")
class Trace(TargetFeature):
    """Enable tracing of all memory accesses in ETISS."""

    DEFAULTS = {**FeatureBase.DEFAULTS, "to_file": True}  # ETISS can only trace to file

    @property
    def to_file(self):
        value = self.config["to_file"]
        return str2bool(value, allow_none=True)

    def __init__(self, features=None, config=None):
        super().__init__("trace", features=features, config=config)

    # def add_target_config(self, target, config, directory=None):
    def add_target_config(self, target, config):
        assert target in ["etiss_pulpino", "etiss", "ovpsim"]
        if target in ["etiss_pulpino", "etiss"]:
            config.update({f"{target}.trace_memory": self.enabled})
        elif target == "ovpsim":
            extra_args_new = config.get("extra_args", [])
            extra_args_new.append("--trace --tracemem SAX")
            if self.to_file:
                # assert directory is not None
                directory = Path(".")  # Need to use relative path because target.dir not available here
                trace_file = directory / "trace.txt"
                extra_args_new.append("--tracefile")
                extra_args_new.append(trace_file)
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
        "use_workspace_io": False,
    }

    def __init__(self, features=None, config=None):
        super().__init__("usmp", features=features, config=config)

    @property
    def algorithm(self):
        return str(self.config["algorithm"])

    @property
    def use_workspace_io(self):
        value = self.config["use_workspace_io"]
        return str2bool(value)

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
        tmp["tir.usmp.use_workspace_io"] = self.use_workspace_io
        if self.algorithm in ["greedy_by_size", "greedy_by_conflicts", "hill_climb"]:
            tmp["tir.usmp.algorithm"] = self.algorithm
        else:
            tmp["tir.usmp.custom_algorithm"] = self.algorithm
        config.update(
            {f"{backend}.extra_pass_config": tmp, f"{backend}.arena_size": 0}
        )  # In recent TVM versions USMP will have it's own arena.

    # -> enable this via backend


@register_feature("fuse_ops")
class FuseOps(BackendFeature):
    """TODO"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "max_depth": 100,
    }

    def __init__(self, features=None, config=None):
        super().__init__("fuse_ops", features=features, config=config)

    @property
    def max_depth(self):
        return int(self.config["max_depth"])

    def add_backend_config(self, backend, config):
        # assert backend in ["tvmaot"], f"Unsupported feature '{self.name}' for backend '{backend}'"
        # TODO: tvm only
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
        tmp["relay.FuseOps.max_depth"] = self.max_depth
        config.update({f"{backend}.extra_pass_config": tmp})


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

    REQUIRED = {"tflite_visualize.exe"}

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
        "mode": "autotvm",  # further options: autoscheduler, metascheduler
        "results_file": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("autotuned", features=features, config=config)

    @property
    def results_file(self):
        return self.config.get("results_file", None)

    @property
    def mode(self):
        value = self.config["mode"]
        assert value in ["autotvm", "autoscheduler", "metascheduler"]
        return value

    def get_backend_config(self, backend):
        assert backend in SUPPORTED_TVM_BACKENDS
        # TODO: error handling her eor on backend?
        return filter_none(
            {
                f"{backend}.use_tuning_results": self.enabled,
                f"{backend}.autotuned_results_file": self.results_file,
                f"{backend}.autotuned_mode": self.mode,
            }
        )


@register_feature("autotune")
class Autotune(RunFeature):
    """Generic autotuning feature for enabling the TUNE stage only."""

    def __init__(self, features=None, config=None):
        super().__init__("autotune", features=features, config=config)

    def get_run_config(self):
        return {"run.tune_enabled": self.enabled}


# not registered!
class TVMTuneBase(PlatformFeature):
    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "results_file": None,
        "append": None,
        "trials": None,
        "trials_single": None,
        "early_stopping": None,
        "num_workers": None,
        "max_parallel": None,
        "use_rpc": None,
        "timeout": None,
        "visualize": None,
        "visualize_file": None,
        "visualize_live": None,
        "tasks": None,
        # All None to use the defaults defined in the backend instead
    }

    @property
    def results_file(self):
        return self.config["results_file"] if "results_file" in self.config else None

    @property
    def append(self):
        return self.config["append"] if "append" in self.config else None

    @property
    def trials(self):
        return self.config["trials"] if "trials" in self.config else None

    @property
    def trials_single(self):
        return self.config["trials_single"] if "trials_single" in self.config else None

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
    def visualize(self):
        return self.config["visualize"]

    @property
    def visualize_file(self):
        return self.config["visualize_file"]

    @property
    def visualize_live(self):
        return self.config["visualize_live"]

    @property
    def tasks(self):
        return self.config["tasks"]

    def get_platform_config(self, platform):
        assert platform in ["tvm", "microtvm"]
        # TODO: figure out a default path automatically
        return filter_none(
            {
                f"{platform}.autotuning_results_file": self.results_file,
                f"{platform}.autotuning_append": self.append,
                f"{platform}.autotuning_trials": self.trials,
                f"{platform}.autotuning_trials_single": self.trials_single,
                f"{platform}.autotuning_early_stopping": self.early_stopping,
                f"{platform}.autotuning_num_workers": self.num_workers,
                f"{platform}.autotuning_max_parallel": self.max_parallel,
                f"{platform}.autotuning_timeout": self.timeout,
                f"{platform}.autotuning_visualize": self.visualize,
                f"{platform}.autotuning_visualize_file": self.visualize_file,
                f"{platform}.autotuning_visualize_live": self.visualize_live,
                f"{platform}.autotuning_tasks": self.tasks,
            }
        )


@register_feature("autotvm", depends=["autotune"])
class AutoTVM(TVMTuneBase):
    """Use the TVM autotuner inside the backend to generate tuning logs."""

    # TODO: graphtuner
    # TODO: tuner base feature class

    DEFAULTS = {
        **TVMTuneBase.DEFAULTS,
        "tuner": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("autotvm", features=features, config=config)

    @property
    def tuner(self):
        return self.config["tuner"] if "tuner" in self.config else None

    def get_platform_config(self, platform):
        ret = super().get_platform_config(platform)
        new = filter_none(
            {
                f"{platform}.autotvm_enable": self.enabled,
                f"{platform}.autotvm_tuner": self.tuner,
            }
        )
        ret.update(new)
        return ret


@register_feature("autoscheduler", depends=["autotune"])
class AutoScheduler(TVMTuneBase):
    """TODO"""

    # TODO: metascheduler
    # TODO: graphtuner
    # TODO: tuner base feature class

    DEFAULTS = {
        **TVMTuneBase.DEFAULTS,
        "include_simple_tasks": None,
        "log_estimated_latency": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("autoschedule", features=features, config=config)

    @property
    def include_simple_tasks(self):
        return self.config["include_simple_tasks"]

    @property
    def log_estimated_latency(self):
        return self.config["log_estimated_latency"]

    def get_platform_config(self, platform):
        ret = super().get_platform_config(platform)
        new = filter_none(
            {
                f"{platform}.autoscheduler_enable": self.enabled,
                f"{platform}.autoscheduler_include_simple_tasks": self.include_simple_tasks,
                f"{platform}.autoscheduler_log_estimated_latency": self.log_estimated_latency,
            }
        )
        ret.update(new)
        return ret


@register_feature("metascheduler", depends=["autotune"])
class MetaScheduler(TVMTuneBase):
    """TODO"""

    DEFAULTS = {
        **TVMTuneBase.DEFAULTS,
    }

    def __init__(self, features=None, config=None):
        super().__init__("metascheduler", features=features, config=config)

    def get_platform_config(self, platform):
        ret = super().get_platform_config(platform)
        new = filter_none(
            {
                f"{platform}.metascheduler_enable": self.enabled,
            }
        )
        ret.update(new)
        return ret


@register_feature("disable_legalize")
class DisableLegalize(BackendFeature, SetupFeature):
    """Enable transformation to reduces sizes of intermediate buffers by skipping legalization passes."""

    REQUIRED = {"tvm_extensions.wrapper"}

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


@register_feature("uma_backends")
class UMABackends(BackendFeature):
    """Add directories that contain UMA backends."""

    REQUIRED = set()

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "uma_dir": "",
        "uma_target": "",
    }

    def __init__(self, features=None, config=None):
        super().__init__("uma_backends", features=features, config=config)

    @property
    def uma_dir(self):
        return self.config["uma_dir"]

    @property
    def uma_target(self):
        return self.config["uma_target"]

    def add_backend_config(self, backend, config):
        assert backend in SUPPORTED_TVM_BACKENDS, f"Unsupported feature '{self.name}' for backend '{backend}'"
        tvmcArgs = ["--experimental-tvmc-extension", self.uma_dir]
        if f"{backend}.tvmc_extra_args" in config:
            config[f"{backend}.tvmc_extra_args"].extend(tvmcArgs)
        else:
            config[f"{backend}.tvmc_extra_args"] = tvmcArgs
        extras = config.get(f"{backend}.extra_target", [])
        if self.uma_target not in extras:
            if isinstance(extras, str):
                extras = [extras]
            extras.append(self.uma_target)
        config[f"{backend}.extra_target"] = extras


@register_feature("demo")
class Demo(PlatformFeature):
    """Run demo application instead of benchmarking code."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "board": None,
        "print_stats": False,
        "print_interval_ms": 5000,
    }

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

    def __init__(self, features=None, config=None):
        super().__init__("cachesim", features=features, config=config)

    @property
    def ic_enable(self):
        value = self.config["ic_enable"]
        return str2bool(value)

    @property
    def ic_config(self):
        return self.config["ic_config"]

    @property
    def dc_enable(self):
        value = self.config["dc_enable"]
        return str2bool(value)

    @property
    def dc_config(self):
        return self.config["dc_config"]

    @property
    def l2_enable(self):
        value = self.config["l2_enable"]
        return str2bool(value)

    @property
    def l2_config(self):
        return self.config["l2_config"]

    @property
    def log_misses(self):
        value = self.config["log_misses"]
        return str2bool(value)

    @property
    def detailed(self):
        value = self.config["detailed"]
        return str2bool(value)

    # def add_target_config(self, target, config, directory=None):
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

            def cachesim_callback(stdout, metrics, artifacts, directory=None):
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

    OPTIONAL = {"etiss.experimental_print_to_file"}

    def __init__(self, features=None, config=None):
        super().__init__("log_instrs", features=features, config=config)

    @property
    def to_file(self):
        value = self.config["to_file"]
        return str2bool(value, allow_none=True)

    @property
    def etiss_experimental_print_to_file(self):
        value = self.config["etiss.experimental_print_to_file"]
        return str2bool(value, allow_none=True)

    # def add_target_config(self, target, config, directory=None):
    def add_target_config(self, target, config):
        assert target in ["spike", "etiss_pulpino", "etiss", "ovpsim", "corev_ovpsim", "gvsoc_pulp"]
        if not self.enabled:
            return
        if target == "spike":
            extra_args_new = config.get("extra_args", [])
            extra_args_new.append("-l")
            if self.to_file:
                # assert directory is not None
                directory = Path(".")  # Need to use relative path because target.dir not available here
                log_file = directory / "instrs.txt"
                extra_args_new.append(f"--log={log_file}")
            config.update({f"{target}.extra_args": extra_args_new})
        elif target in ["etiss_pulpino", "etiss"]:
            plugins_new = config.get("plugins", [])
            plugins_new.append("PrintInstruction")
            config.update({f"{target}.plugins": plugins_new})
            if self.etiss_experimental_print_to_file:
                extra_bool_config_new = config.get("extra_bool_config", {})
                if self.to_file:
                    extra_bool_config_new["plugin.printinstruction.print_to_file"] = True
                config.update({f"{target}.extra_bool_config": extra_bool_config_new})
        elif target in ["ovpsim", "corev_ovpsim"]:
            extra_args_new = config.get("extra_args", [])
            extra_args_new.append("--trace")
            if self.to_file:
                # assert directory is not None
                directory = Path(".")  # Need to use relative path because target.dir not available here
                log_file = directory / "instrs.txt"
                extra_args_new.append("--tracefile")
                extra_args_new.append(log_file)
            config.update({f"{target}.extra_args": extra_args_new})
        elif target == "gvsoc_pulp":
            extra_args_new = config.get("extra_args", [])
            if self.to_file:
                # assert directory is not None
                directory = Path(".")  # Need to use relative path because target.dir not available here
                log_file = directory / "instrs.txt"
                extra_args_new.append(f"--trace=insn:{log_file}")
            else:
                extra_args_new.append("--trace=insn")
            config.update({f"{target}.extra_args": extra_args_new})

    def get_target_callbacks(self, target):
        assert target in [
            "spike",
            "etiss_pulpino",
            "etiss",
            "ovpsim",
            "corev_ovpsim",
            "gvsoc_pulp",
        ], f"Unsupported feature '{self.name}' for target '{target}'"
        if self.enabled:
            if not target == "gvsoc_pulp":

                def log_instrs_callback(stdout, metrics, artifacts, directory=None):
                    """Callback which parses the targets output and updates the generated metrics and artifacts."""
                    new_lines = []
                    if self.to_file:
                        if target in ["etiss_pulpino", "etiss"]:
                            if self.etiss_experimental_print_to_file:
                                log_file = Path(directory) / "instr_trace.csv"
                            else:
                                # TODO: update stdout and remove log_instrs lines
                                instrs = []
                                for line in stdout.split("\n"):
                                    if target in ["etiss_pulpino", "etiss"]:
                                        expr = re.compile(r"0x[a-fA-F0-9]+: .* \[.*\]")
                                    match = expr.match(line)
                                    if match is not None:
                                        instrs.append(line)
                                    else:
                                        new_lines.append(line)
                                content = "\n".join(instrs)
                                stdout = "\n".join(new_lines)
                        else:
                            assert target in ["spike", "ovpsim", "corev_ovpsim"]
                            log_file = Path(directory) / "instrs.txt"
                            with open(log_file, "r") as f:
                                content = f.read()
                        instrs_artifact = Artifact(
                            f"{target}_instrs.log",
                            content=content,
                            fmt=ArtifactFormat.TEXT,
                            flags=(self.name, target),
                        )
                        artifacts.append(instrs_artifact)
                    return stdout

                return None, log_instrs_callback
        return None, None


@register_feature("arm_mvei", depends=["arm_dsp"])
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
        "layouts": True,
        "schedules": True,
    }

    @property
    def layouts(self):
        value = self.config["layouts"]
        return str2bool(value)

    @property
    def schedules(self):
        value = self.config["schedules"]
        return str2bool(value)

    def __init__(self, features=None, config=None):
        super().__init__("target_optimized", features=features, config=config)

    def get_run_config(self):
        if self.enabled:
            return {
                "run.target_optimized_layouts": self.layouts,
                "run.target_optimized_schedules": self.schedules,
            }
        else:
            return {}


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
        "force_vector_width": None,  # llvm only
        "force_vector_interleave": None,  # llvm only
        "custom_unroll": False,  # TODO: this is not related to vectorization -> move to llvm toolchain!
    }

    def __init__(self, features=None, config=None):
        super().__init__("auto_vectorize", features=features, config=config)

    @property
    def verbose(self):
        value = self.config["verbose"]
        # return str2bool(value)
        if value is None or not value:
            return "OFF"
        assert isinstance(value, str)
        value = value.lower()
        assert value in ["loop", "slp", "none"]
        return value

    @property
    def loop(self):
        value = self.config["loop"]
        return str2bool(value)

    @property
    def slp(self):
        value = self.config["slp"]
        return str2bool(value)

    @property
    def force_vector_width(self):
        value = self.config["force_vector_width"]
        # bool not allowed!
        if value is None:
            return "OFF"
        if isinstance(value, str):
            value = int(value)
        assert isinstance(value, int)
        if value <= 1:
            return "OFF"
        return value

    @property
    def force_vector_interleave(self):
        value = self.config["force_vector_interleave"]
        # bool not allowed!
        if value is None:
            return "OFF"
        if isinstance(value, str):
            value = int(value)
        assert isinstance(value, int)
        if value <= 1:
            return "OFF"
        return value

    @property
    def custom_unroll(self):
        value = self.config["custom_unroll"]
        return str2bool(value)

    def get_platform_defs(self, platform):
        return {
            "RISCV_AUTO_VECTORIZE": self.enabled,
            "RISCV_AUTO_VECTORIZE_VERBOSE": self.verbose,
            "RISCV_AUTO_VECTORIZE_LOOP": self.loop and self.enabled,
            "RISCV_AUTO_VECTORIZE_SLP": self.slp and self.enabled,
            "RISCV_AUTO_VECTORIZE_FORCE_VECTOR_WIDTH": self.force_vector_width,
            "RISCV_AUTO_VECTORIZE_FORCE_VECTOR_INTERLEAVE": self.force_vector_interleave,
            "RISCV_AUTO_VECTORIZE_CUSTOM_UNROLL": self.custom_unroll,
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
        return str2bool(value)

    @property
    def aggregate(self):
        value = self.config["aggregate"]
        assert value in ["avg", "all", "max", "min", "none"]
        return value

    def get_platform_config(self, platform):
        supported = ["mlif", "tvm", "microtvm"]  # TODO: support espidf
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

            def benchmark_callback(stdout, metrics, artifacts, directory=None):
                if len(metrics) <= 1:
                    return stdout
                metrics_ = metrics[1:]  # drop first run (warmup)

                # TODO: this currently processes all numeric metrics, should probably ignore stuff like MIPS etc.
                candidates = ["cycle", "time", "instruction"]  # TODO: allow overriding via config
                data_ = [
                    {
                        key: (float(value) / self.num_runs) if self.num_runs > 1 else value
                        for key, value in m.data.items()
                        if any(x in key.lower() for x in candidates)
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

                if len(df.columns) == 0:
                    data = {}
                elif len(aggs) == 0:
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

    def __init__(self, features=None, config=None):
        super().__init__("tvm_profile", features=features, config=config)

    def get_platform_config(self, platform):
        supported = ["tvm", "microtvm"]
        assert platform in supported, f"Unsupported feature '{self.name}' for platform '{platform}'"
        return {
            f"{platform}.profile": self.enabled,
        }


@register_feature("xcorev")
class XCoreV(TargetFeature, PlatformFeature, SetupFeature):
    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "mac": True,
        "mem": True,
        "bi": True,
        "alu": True,
        "bitmanip": True,
        "simd": True,
        "hwlp": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("xcorev", features=features, config=config)

    @property
    def mac(self):
        value = self.config["mac"]
        return str2bool(value)

    @property
    def mem(self):
        value = self.config["mem"]
        return str2bool(value)

    @property
    def bi(self):
        value = self.config["bi"]
        return str2bool(value)

    @property
    def alu(self):
        value = self.config["alu"]
        return str2bool(value)

    @property
    def bitmanip(self):
        value = self.config["bitmanip"]
        return str2bool(value)

    @property
    def simd(self):
        value = self.config["simd"]
        return str2bool(value)

    @property
    def hwlp(self):
        value = self.config["hwlp"]
        return str2bool(value)

    # def add_target_config(self, target, config, directory=None):
    def add_target_config(self, target, config):
        assert target in [
            "etiss",
            "microtvm_etiss",
            "corev_ovpsim",
            "cv32e40p",
        ], f"Unsupported feature '{self.name}' for target '{target}'"
        if self.enabled:
            config[f"{target}.enable_xcorevmac"] = self.mac
            config[f"{target}.enable_xcorevmem"] = self.mem
            config[f"{target}.enable_xcorevbi"] = self.bi
            config[f"{target}.enable_xcorevalu"] = self.alu
            config[f"{target}.enable_xcorevbitmanip"] = self.bitmanip
            config[f"{target}.enable_xcorevsimd"] = self.simd
            config[f"{target}.enable_xcorevhwlp"] = self.hwlp


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

    REQUIRED = {"pulp_gcc.install_dir", "pulp_gcc.name"}

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
    def generalized_str2bool(inp: Union[str, bool, int]) -> bool:
        return str2bool(inp)

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
            "EXTRA_C_FLAGS": EXTRA_FLAGS,
            "EXTRA_CXX_FLAGS": EXTRA_FLAGS,
            "EXTRA_ASM_FLAGS": EXTRA_FLAGS,
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
                if isinstance(dict1_value, (str, list)) and type(dict1_value) is type(dict2_value):
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

    REQUIRED = {"tflite_pack.exe"}

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


@register_feature("tflite_analyze")
class TfLiteAnalyze(FrontendFeature):
    """Get the estimated ROM, RAM and MACs from a TFLite model."""

    REQUIRED = {"tflite_analyze.exe"}

    def __init__(self, features=None, config=None):
        super().__init__("tflite_analyze", features=features, config=config)

    @property
    def tflite_analyze_exe(self):
        return self.config["tflite_analyze.exe"]

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name}' for frontend '{frontend}'"
        return filter_none(
            {
                f"{frontend}.analyze_enable": self.enabled,
                f"{frontend}.analyze_script": self.tflite_analyze_exe,
            }
        )

    def update_formats(self, frontend, input_formats, output_formats):
        assert frontend in ["tflite"], f"Unsupported feature '{self.name}' for frontend '{frontend}'"
        if self.enabled:
            output_formats.append(ArtifactFormat.TEXT)


# @register_feature("hpmcounter")
class HpmCounter(TargetFeature, PlatformFeature):  # TODO: SetupFeature?
    """Use RISC-V Performance Counters"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "num_counters": 32,
        "supported_counters": 1,  # To check if number of enabled counters exceeds counters implemented in hw
        "enabled_counters": [],
        "counter_names": [],
    }

    # def __init__(self, features=None, config=None):
    #     super().__init__("hpmcounter", features=features, config=config)

    @property
    def num_counters(self):
        temp = self.config["num_counters"]
        return int(temp)

    @property
    def supported_counters(self):
        temp = self.config["supported_counters"]
        return int(temp)

    @property
    def enabled_counters(self):
        temp = self.config["enabled_counters"]
        if isinstance(temp, int):
            temp = [temp]
        elif isinstance(temp, str):
            temp = str2list(temp)
        assert isinstance(temp, list)
        temp = list(map(int, temp))
        return temp

    @property
    def counter_names(self):
        temp = self.config["counter_names"]
        if not isinstance(temp, list):
            temp = str2list(temp)
        return temp

    def get_platform_defs(self, platform):
        assert platform in ["mlif"], f"Unsupported feature '{self.name}' for platform '{platform}'"
        assert self.supported_counters >= len(self.enabled_counters)
        assert max(self.enabled_counter) < self.supported_counters
        return {
            "HPM_COUNTERS": self.num_counters,
            **({f"USE_HPM{i}": i in self.enabled_counters for i in range(self.num_counters)}),
        }

    def get_target_callbacks(self, target):
        if self.enabled:

            def hpm_callback(stdout, metrics, artifacts, directory=None):
                """Callback for extracting HPM metrics from stdout"""
                print("stdout", stdout)
                # TODO: add metrics
                # TODO: remove HPM lines from stdout
                return stdout

            return None, hpm_callback
        return None, None


@register_feature("cv32_hpmcounter")
class CV32HpmCounter(HpmCounter):  # TODO: SetupFeature?
    """Use RISC-V Performance Counters"""

    DEFAULTS = {
        **HpmCounter.DEFAULTS,
        "num_counters": 12,
        "supported_counters": 32,  # TODO
        "enabled_counters": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "counter_names": [
            "Cycles",
            "Instructions",
            "LD Stalls",
            "JMP Stalls",
            "IMiss",
            "LD",
            "ST",
            "Jump",
            "Branch",
            "Branch Taken",
            "Compressed",
            "Pipe Stall",
        ],
    }

    def __init__(self, features=None, config=None):
        super().__init__("cv32_hpmcounter", features=features, config=config)


@register_feature("vanilla_accelerator")
class VanillaAccelerator(TargetFeature):
    """TODO"""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "plugin_name": "VanillaAccelerator",
        "base_addr": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("vanilla_accelerator", features=features, config=config)

    @property
    def plugin_name(self):
        value = self.config["plugin_name"]
        return value

    @property
    def base_addr(self):
        value = self.config["base_addr"]
        return value

    def add_target_config(self, target, config):
        assert target in ["etiss"]
        if not self.enabled:
            return
        plugins_new = config.get(f"{target}.plugins", [])
        plugins_new.append(self.plugin_name)
        config.update({f"{target}.plugins": plugins_new})
        if self.base_addr is not None:
            extra_plugin_config = config.get(f"{target}.extra_plugin_config", {})
            assert self.name not in extra_plugin_config
            extra_plugin_config[self.name]["baseaddr"] = self.base_addr
            config.update({f"{target}.extra_plugin_config": extra_plugin_config})


@register_feature("gen_data")
class GenData(FrontendFeature):  # TODO: use custom stage instead of LOAD
    """Generate input data for validation."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "fill_mode": "file",  # Allowed: random, ones, zeros, file, dataset
        "file": "auto",  # Only relevant if fill_mode=file
        "number": 10,  # generate up to number samples (may be less if file has less inputs)
        "fmt": "npy",  # Allowed: npy, npz
    }

    def __init__(self, features=None, config=None):
        super().__init__("gen_data", features=features, config=config)

    @property
    def fill_mode(self):
        value = self.config["fill_mode"]
        assert value in ["random", "ones", "zeros", "file", "dataset"]
        return value

    @property
    def file(self):
        value = self.config["file"]
        return value

    @property
    def number(self):
        return int(self.config["number"])

    @property
    def fmt(self):
        value = self.config["fmt"]
        assert value in ["npy", "npz"]
        return value

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"]
        return {
            f"{frontend}.gen_data": self.enabled,
            f"{frontend}.gen_data_fill_mode": self.fill_mode,
            f"{frontend}.gen_data_file": self.file,
            f"{frontend}.gen_data_number": self.number,
            f"{frontend}.gen_data_fmt": self.fmt,
        }


@register_feature("gen_ref_data", depends=["gen_data"])
class GenRefData(FrontendFeature):  # TODO: use custom stage instead of LOAD
    """Generate reference outputs for validation."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "mode": "file",  # Allowed: file, model
        "file": "auto",  # Only relevant if mode=file
        "fmt": "npy",  # Allowed: npy, npz
    }

    def __init__(self, features=None, config=None):
        super().__init__("gen_ref_data", features=features, config=config)

    @property
    def mode(self):
        value = self.config["mode"]
        assert value in ["file", "model"]
        return value

    @property
    def file(self):
        value = self.config["file"]
        return value

    @property
    def fmt(self):
        value = self.config["fmt"]
        assert value in ["npy", "npz"]
        return value

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"]
        return {
            f"{frontend}.gen_ref_data": self.enabled,
            f"{frontend}.gen_ref_data_mode": self.mode,
            f"{frontend}.gen_ref_data_file": self.file,
            f"{frontend}.gen_ref_data_fmt": self.fmt,
        }


@register_feature("gen_ref_labels", depends=["gen_data"])
class GenRefLabels(FrontendFeature):  # TODO: use custom stage instead of LOAD
    """Generate reference labels for classification."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "mode": "file",  # Allowed: file, model
        "file": "auto",  # Only relevant if mode=file
        "fmt": "npy",  # Allowed: npy, npz, txt
    }

    def __init__(self, features=None, config=None):
        super().__init__("gen_ref_labels", features=features, config=config)

    @property
    def mode(self):
        value = self.config["mode"]
        assert value in ["file", "model"]
        return value

    @property
    def file(self):
        value = self.config["file"]
        return value

    @property
    def fmt(self):
        value = self.config["fmt"]
        assert value in ["npy", "npz"]
        return value

    def get_frontend_config(self, frontend):
        assert frontend in ["tflite"]
        return {
            f"{frontend}.gen_ref_labels": self.enabled,
            f"{frontend}.gen_ref_labels_mode": self.mode,
            f"{frontend}.gen_ref_labels_file": self.file,
            f"{frontend}.gen_ref_labels_fmt": self.fmt,
        }


@register_feature("set_inputs")
class SetInputs(PlatformFeature):  # TODO: use custom stage instead of LOAD
    """Apply test inputs to model."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "interface": "auto",  # Allowed: auto, rom, filesystem, stdin, stdin_raw, uart
    }

    def __init__(self, features=None, config=None):
        super().__init__("set_inputs", features=features, config=config)

    @property
    def interface(self):
        value = self.config["interface"]
        assert value in ["auto", "rom", "filesystem", "stdin", "stdin_raw", "uart"]
        return value

    def get_platform_config(self, platform):
        assert platform in ["mlif", "tvm", "microtvm"]
        # if tvm/microtvm: allow using --fill-mode provided by tvmc run
        return {
            f"{platform}.set_inputs": self.enabled,
            f"{platform}.set_inputs_interface": self.interface,
        }


@register_feature("get_outputs")
class GetOutputs(PlatformFeature):  # TODO: use custom stage instead of LOAD
    """Extract resulting outputs from model."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
        "interface": "auto",  # Allowed: auto, filesystem, stdout, stdout_raw, uart
        "fmt": "npy",  # Allowed: npz, npz
    }

    def __init__(self, features=None, config=None):
        super().__init__("get_outputs", features=features, config=config)

    @property
    def interface(self):
        value = self.config["interface"]
        assert value in ["auto", "filesystem", "stdout", "stdout_raw", "uart"]
        return value

    @property
    def fmt(self):
        value = self.config["fmt"]
        assert value in ["npy", "npz"]
        return value

    def get_platform_config(self, platform):
        assert platform in ["mlif", "tvm", "microtvm"]
        return {
            f"{platform}.get_outputs": self.enabled,
            f"{platform}.get_outputs_interface": self.interface,
            f"{platform}.get_outputs_fmt": self.fmt,
        }


@register_feature("validate_new", depends=["gen_data", "gen_ref_data", "set_inputs", "get_outputs"])
class ValidateNew(RunFeature):
    """Wrapper feature for enabling all validatioon related features at once."""

    DEFAULTS = {
        **FeatureBase.DEFAULTS,
    }

    def __init__(self, features=None, config=None):
        super().__init__("validate_new", features=features, config=config)

    # def get_postprocesses(self):
    #     # config = {}
    #     # from mlonmcu.session.postprocess import ValidateOutputsPostprocess
    #     # validate_outputs_postprocess = ValidateOutputsPostprocess(features=[], config=config)
    #     # return [validate_outputs_postprocess]
    #     return ["validate_outputs"]
