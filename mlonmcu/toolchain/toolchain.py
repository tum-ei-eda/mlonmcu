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
from abc import ABC
from pathlib import Path

from mlonmcu.feature.features import get_matching_features
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.config import filter_config, str2bool
from mlonmcu.utils import filter_none

from mlonmcu.logging import get_logger


logger = get_logger()


class Toolchain(ABC):
    FEATURES = set()

    DEFAULTS = {}

    REQUIRED = set()
    OPTIONAL = set()

    def __init__(self, name, features=None, config=None):
        self.name = name
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.OPTIONAL, self.REQUIRED)

    def __repr__(self):
        probs = []
        if self.name:
            probs.append(self.name)
        if self.features and len(self.features) > 0:
            probs.append(str(self.features))
        if self.config and len(self.config) > 0:
            probs.append(str(self.config))
        return "Toolchain(" + ",".join(probs) + ")"

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.TOOLCHAIN)
        for feature in features:
            assert (  # If this assertion occurs, continue with the next toolchain instead of failing
                # (TODO: create custom exception type)
                feature.name
                in self.FEATURES
            ), f"Incompatible feature: {feature.name}"
            # Instead we might introduce self.compatible and set it to true at this line
            feature.used = True
            feature.add_toolchain_config(self.name, self.config)
        return features

    @property
    def supported_platforms(self):
        raise NotImplementedError

    @property
    def supported_architectures(self):
        raise NotImplementedError

    def get_platform_defs(self, platform):
        return {}

    def add_platform_defs(self, platform, defs):
        defs.update(self.get_platform_defs(platform))

    def get_backend_config(self, backend):
        return {}

    def add_backend_config(self, backend, config):
        new = filter_none(
            self.get_backend_config(
                backend
            )
        )

        # only allow overwriting non-none values
        # to support accepting user-vars
        new = {key: value for key, value in new.items() if config.get(key, None) is None}
        config.update(new)


class MlifToolchain(Toolchain):
    """TODO"""

    def __init__(self, name, features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def supported_platforms(self):
        return ["mlif"]

    @property
    def supported_architectures(self):
        raise NotImplementedError


class NoneToolchain(Toolchain):
    """TODO"""

    def __init__(self, name="none", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def supported_platforms(self):
        return []

    @property
    def supported_architectures(self):
        return []


class DefaultToolchain(MlifToolchain):  # Fallback
    """TODO"""

    def __init__(self, name="default", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def supported_architectures(self):
        return ["riscv", "riscv32", "riscv64", "arm", "x86", "x86_64"]


class GCCToolchain(MlifToolchain):
    """TODO"""

    FEATURES = MlifToolchain.FEATURES | {"auto_vectorize"}

    def __init__(self, name="gcc", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def supported_architectures(self):
        return ["riscv", "riscv32", "riscv64", "arm", "x86", "x86_64"]

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        # TODO refactor the following using inheritance instead of branching
        ret["TOOLCHAIN"] = "gcc"
        return ret


class LLVMToolchain(MlifToolchain):
    """TODO"""

    FEATURES = MlifToolchain.FEATURES | {"auto_vectorize"}

    REQUIRED = {"llvm.install_dir"}

    def __init__(self, name="llvm", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def llvm_install_dir(self):
        return Path(self.config[f"{self.name}.install_dir"])

    @property
    def supported_architectures(self):
        return ["riscv", "riscv32", "riscv64", "arm", "x86", "x86_64"]

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        # TODO refactor the following using inheritance instead of branching
        ret["TOOLCHAIN"] = "llvm"
        ret["LLVM_DIR"] = self.llvm_install_dir
        return ret


class RISCVGCCToolchain(GCCToolchain):
    """TODO"""

    DEFAULTS = {
        "arch": None,
        "abi": None,
        "attr": None,
    }

    REQUIRED = GCCToolchain.REQUIRED | {"riscv_gcc.install_dir", "riscv_gcc.name"}

    def __init__(self, name="riscv_gcc", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def riscv_gcc_prefix(self):
        return Path(self.config[f"{self.name}.install_dir"])

    @property
    def riscv_gcc_basename(self):
        return Path(self.config[f"{self.name}.name"])

    @property
    def arch(self):
        value = self.config["arch"]
        return value

    @property
    def abi(self):
        value = self.config["abi"]
        return value

    @property
    def attr(self):
        value = self.config["attr"]
        return value

    @property
    def xlen(self):
        assert self.arch is not None
        ret = int(self.arch[2:4])
        return ret

    @property
    def supported_architectures(self):
        return ["riscv", "riscv32", "riscv64"]

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        # TODO refactor the following using inheritance instead of branching
        ret["RISCV_ELF_GCC_PREFIX"] = self.riscv_gcc_prefix
        ret["RISCV_ELF_GCC_BASENAME"] = self.riscv_gcc_basename
        if self.arch:
            ret["RISCV_ARCH"] = self.arch
        if self.abi:
            ret["RISCV_ABI"] = self.abi
        # if self.attr:
        #     ret["RISCV_ATTR"] = self.attr  # TODO: use for clang
        return ret

    def get_backend_config(self, backend):
        if backend in SUPPORTED_TVM_BACKENDS:
            ret = {
                "target_march": self.arch,
                "target_mtriple": self.riscv_gcc_basename,  # TODO: riscv32-esp-elf for esp32c3!
                "target_mabi": self.abi,
                "target_mattr": self.attr,
                "target_mcpu": f"generic-rv{self.xlen}",
            }
            return ret
        return {}


class RISCVLLVMToolchain(LLVMToolchain):
    """TODO"""

    FEATURES = LLVMToolchain.FEATURES | {"vext"}

    DEFAULTS = {
        "arch": None,
        "abi": None,
        "attr": None,
    }

    REQUIRED = LLVMToolchain.REQUIRED | {"riscv_gcc.install_dir", "riscv_gcc.name"}

    def __init__(self, name="riscv_llvm", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def llvm_install_dir(self):
        return Path(self.config[f"llvm.install_dir"])

    @property
    def riscv_gcc_prefix(self):
        return Path(self.config["riscv_gcc.install_dir"])

    @property
    def riscv_gcc_basename(self):
        return Path(self.config["riscv_gcc.name"])

    @property
    def arch(self):
        value = self.config["arch"]
        return value

    @property
    def abi(self):
        value = self.config["abi"]
        return value

    @property
    def attr(self):
        value = self.config["attr"]
        return value

    @property
    def xlen(self):
        assert self.arch is not None
        ret = int(self.arch[2:4])
        return ret

    @property
    def supported_architectures(self):
        return ["riscv", "riscv32", "riscv64"]

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        # TODO refactor the following using inheritance instead of branching
        ret["RISCV_ELF_GCC_PREFIX"] = self.riscv_gcc_prefix
        ret["RISCV_ELF_GCC_BASENAME"] = self.riscv_gcc_basename
        if self.arch:
            ret["RISCV_ARCH"] = self.arch
        if self.abi:
            ret["RISCV_ABI"] = self.abi
        # if self.attr:
        #     ret["RISCV_ATTR"] = self.attr  # TODO: use for clang
        return ret

    def get_backend_config(self, backend):
        if backend in SUPPORTED_TVM_BACKENDS:
            ret = {
                "target_march": self.arch,
                "target_mtriple": self.riscv_gcc_basename,  # TODO: riscv32-esp-elf for esp32c3!
                "target_mabi": self.abi,
                "target_mattr": self.attr,
                "target_mcpu": f"generic-rv{self.xlen}",
            }
            return ret
        return {}


class RVVGCCToolchain(RISCVGCCToolchain):
    """TODO"""

    FEATURES = RISCVGCCToolchain.FEATURES | {"vext"}

    REQUIRED = {"riscv_gcc_vext.install_dir", "riscv_gcc_vext.name"}

    def __init__(self, name="riscv_gcc_vext", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if platform == "mlif":
            if self.enable_vext:
                ret["RISCV_VEXT"] = self.enable_vext
            if self.vlen:
                ret["RISCV_VLEN"] = self.vlen
        return ret


class RVPGCCToolchain(RISCVGCCToolchain):
    """TODO"""

    FEATURES = RISCVGCCToolchain.FEATURES | {"pext"}

    REQUIRED = {"riscv_gcc_pext.install_dir", "riscv_gcc_pext.name"}

    def __init__(self, name="riscv_gcc_pext", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )


class XuantieRISCVGCCToolchain(RISCVGCCToolchain):
    """TODO"""

    FEATURES = RISCVGCCToolchain.FEATURES | {"pext", "vext", "xthead"}

    REQUIRED = {"xuantie_riscv_gcc.install_dir", "xuantie_riscv_gcc.name"}

    def __init__(self, name="xuantie_riscv_gcc", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )


class PulpRISCVGCCToolchain(RISCVGCCToolchain):
    """TODO"""

    FEATURES = RISCVGCCToolchain.FEATURES | {"xpulpnn"}

    REQUIRED = {"pulp_riscv_gcc.install_dir", "pulp_riscv_gcc.name"}

    def __init__(self, name="pulp_riscv_gcc", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        # TODO refactor the following using inheritance instead of branching
        ret["RISCV_ELF_GCC_PREFIX"] = self.riscv_gcc_prefix
        ret["RISCV_ELF_GCC_BASENAME"] = self.riscv_gcc_basename
        return ret


class PulpLLVMToolchain(RISCVLLVMToolchain):
    """TODO"""

    FEATURES = RISCVLLVMToolchain.FEATURES | {"xpulpnn"}

    REQUIRED = {"pulp_llvm.install_dir"} | PulpRISCVGCCToolchain.REQUIRED

    def __init__(self, name="pulp_llvm", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )


class CoreVRISCVGCCToolchain(RISCVGCCToolchain):
    """TODO"""

    FEATURES = RISCVGCCToolchain.FEATURES | {"xcorev"}

    REQUIRED = {"corev_riscv_gcc.install_dir", "corev_riscv_gcc.name"}

    def __init__(self, name="corev_riscv_gcc", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )


class CoreVLLVMToolchain(RISCVLLVMToolchain):
    """TODO"""

    FEATURES = RISCVLLVMToolchain.FEATURES | {"xcorev"}

    REQUIRED = {"corev_llvm.install_dir"} | CoreVRISCVGCCToolchain.REQUIRED

    def __init__(self, name="corev_llvm", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )


class ARMGCCToolchain(GCCToolchain):
    """TODO"""

    FEATURES = GCCToolchain.FEATURES | {"arm_dsp", "arm_mvei"}

    DEFAULTS = {
        "cpu": None,
        "float_abi": None,
        "fpu": None,
        "attr": None,
    }

    REQUIRED = {"arm_gcc.install_dir"}

    def __init__(self, name="arm_gcc", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def arm_gcc_prefix(self):
        return str(self.config["arm_gcc.install_dir"])

    @property
    def supported_architectures(self):
        return ["arm"]

    @property
    def cpu(self):
        value = self.config["cpu"]
        return value

    @property
    def float_abi(self):
        value = self.config["float_abi"]
        return value

    @property
    def fpu(self):
        value = self.config["fpu"]
        return value

    @property
    def attr(self):
        value = self.config["attr"]
        return value

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["ARM_COMPILER_PREFIX"] = self.arm_gcc_prefix
        ret["ARM_CPU"] = self.cpu
        ret["ARM_FLOAT_ABI"] = self.float_abi
        ret["ARM_FPU"] = self.fpu
        return ret

    def get_backend_config(self, backend):
        ret = {}
        if backend in SUPPORTED_TVM_BACKENDS:
            ret = {
                # "target_march": self.get_arch(),
                "target_mtriple": "arm-none-eabi",
                "target_mcpu": self.cpu,
                # "target_mattr": "?",
                # "target_mabi": self.float_abi,
                "target_model": f"{self.name}-{self.cpu}",
            }
        return ret


class ARMLLVMToolchain(LLVMToolchain):
    """TODO"""

    FEATURES = GCCToolchain.FEATURES | {"arm_dsp", "arm_mvei"}

    REQUIRED = set()  # TODO

    def __init__(self, name="arm_llvm", features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )

    @property
    def supported_architectures(self):
        return ["arm"]
