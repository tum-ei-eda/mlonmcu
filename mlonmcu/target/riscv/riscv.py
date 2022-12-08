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
"""MLonMCU RISC-V Target definitions"""

from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target import Target
from .util import sort_extensions_canonical, join_extensions

logger = get_logger()


class RISCVTarget(Target):
    """Common base class for RISCV-like targets. Please do not use this as a target itself!"""

    FEATURES = Target.FEATURES + []

    DEFAULTS = {
        **Target.DEFAULTS,
        "xlen": 32,
        "extensions": ["g", "c"],  # TODO overwrite extensions elegantly
        "timeout_sec": 0,  # disabled
        "extra_args": "",
        "fpu": "double",  # allowed: none, single, double
        "arch": None,
        "abi": None,
        "attr": "",
    }
    REQUIRED = ["riscv_gcc.install_dir", "riscv_gcc.name", "riscv_gcc.variant"]
    PUPL_GCC_TOOLCHAIN_REQUIRED = ["pulp_gcc.install_dir", "pulp_gcc.name"]
    OPTIONAL = ["llvm.install_dir"]

    @property
    def riscv_gcc_prefix(self):
        return Path(self.config["riscv_gcc.install_dir"])

    @property
    def riscv_gcc_basename(self):
        return Path(self.config["riscv_gcc.name"])

    @property
    def pulp_gcc_prefix(self):
        return Path(self.config["pulp_gcc.install_dir"])

    @property
    def pulp_gcc_basename(self):
        return Path(self.config["pulp_gcc.name"])

    @property
    def gcc_variant(self):
        return self.config["riscv_gcc.variant"]

    @property
    def xlen(self):
        return int(self.config["xlen"])

    @property
    def extensions(self):
        exts = self.config.get("extensions", []).copy()
        if not isinstance(exts, list):
            exts = exts.split(",")
        if "g" not in exts:
            required = []
            if self.fpu == "double":
                required.append("d")
                required.append("f")
            if self.fpu == "single":
                required.append("f")
            for ext in required:
                if ext not in exts:
                    exts.append(ext)
        return exts

    @property
    def arch(self):
        temp = self.config["arch"]  # TODO: allow underscores and versions
        if temp:
            return temp
        else:
            exts_str = join_extensions(sort_extensions_canonical(self.extensions, lower=True))
            return f"rv{self.xlen}{exts_str}"

    @property
    def abi(self):
        temp = self.config["abi"]
        if temp:
            return temp
        else:
            if self.xlen == 32:
                temp = "ilp32"
            elif self.xlen == 64:
                temp = "lp64"
            else:
                raise RuntimeError(f"Invalid xlen: {self.xlen}")
            if "d" in self.extensions or "g" in self.extensions:
                temp += "d"
            elif "f" in self.extensions:
                temp += "f"
            return temp

    @property
    def attr(self):
        attrs = str(self.config["attr"]).split(",")
        if len(attrs) == 1 and len(attrs[0]) == 0:
            attrs = []
        for ext in sort_extensions_canonical(self.extensions, lower=True, unpack=True):
            attrs.append(f"+{ext}")
        attrs = list(set(attrs))
        return ",".join(attrs)

    @property
    def extra_args(self):
        ret = self.config["extra_args"]
        if isinstance(ret, str):
            if len(ret) == 0:
                ret = []
            else:
                ret = [ret]  # TODO: properly split quoted args
        return ret

    @property
    def timeout_sec(self):
        return int(self.config["timeout_sec"])

    @property
    def fpu(self):
        value = self.config["fpu"]
        if value is None or not value:
            value = "none"
        assert value in ["none", "single", "double"]
        return value

    @property
    def has_fpu(self):
        return self.fpu != "none"

    def get_target_system(self):
        return "generic_riscv"  # TODO: rename to generic-rv32 for compatibility with LLVM

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if "riscv_gcc.install_dir" in self.REQUIRED:  # the target chooses to use the riscv_gcc toolchain
            ret["RISCV_ELF_GCC_PREFIX"] = self.riscv_gcc_prefix
            ret["RISCV_ELF_GCC_BASENAME"] = self.riscv_gcc_basename
        elif "pulp_gcc.install_dir" in self.REQUIRED:  # the target chooses to use the pulp_gcc toolchain
            ret["RISCV_ELF_GCC_PREFIX"] = self.pulp_gcc_prefix
            ret["RISCV_ELF_GCC_BASENAME"] = self.pulp_gcc_basename
        ret["RISCV_ARCH"] = self.arch
        ret["RISCV_ABI"] = self.abi
        ret["RISCV_ATTR"] = self.attr  # TODO: use for clang
        return ret

    def get_arch(self):
        return "riscv"

    def get_backend_config(self, backend):
        if backend in SUPPORTED_TVM_BACKENDS:
            return {
                "target_device": "riscv_cpu",
                "target_march": self.arch,
                "target_model": "unknown",
                "target_mtriple": self.riscv_gcc_basename,  # TODO: riscv32-esp-elf for esp32c3!
                "target_mabi": self.abi,
                "target_mattr": self.attr,
                "target_mcpu": f"generic-rv{self.xlen}",
            }
        return {}
