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
from .target import Target

logger = get_logger()


def sort_extensions_canonical(extensions):
    """Utility to get the canonical architecture name string."""

    # See: https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf#table.22.1
    ORDER = [
        "I",
        "M",
        "A",
        "F",
        "D",
        "G",
        "Q",
        "L",
        "C",
        "B",
        "J",
        "T",
        "P",
        "V",
        "X",
        "S",
        "SX",
    ]  # What about Z* extensions?
    extensions_new = extensions.copy()

    def _get_index(x):
        if x in ORDER:
            return ORDER.index(x)
        else:
            for i, o in enumerate(ORDER):
                if x.startswith(o):
                    return i
            return ORDER.index("X") - 0.5  # Insert unknown keys right before custom extensions

    extensions_new.sort(key=lambda x: _get_index(x))
    return extensions_new


class RISCVTarget(Target):
    """Common base class for RISCV-like targets. Please do not use this as a target itself!"""

    FEATURES = []

    DEFAULTS = {
        **Target.DEFAULTS,
        "timeout_sec": 0,  # disabled
        "extra_args": "",
        "arch": "rv32gc",
        "abi": "ilp32d",
    }
    REQUIRED = ["riscv_gcc.install_dir", "riscv_gcc.name"]

    @property
    def riscv_prefix(self):
        return Path(self.config["riscv_gcc.install_dir"])

    @property
    def riscv_basename(self):
        return Path(self.config["riscv_gcc.name"])

    @property
    def arch(self):
        return str(self.config["arch"])

    @property
    def abi(self):
        return str(self.config["abi"])

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

    def get_target_system(self):
        return "generic_riscv"

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["RISCV_ELF_GCC_PREFIX"] = self.riscv_prefix
        ret["RISCV_ELF_GCC_BASENAME"] = self.riscv_basename
        ret["RISCV_ARCH"] = self.arch
        ret["RISCV_ABI"] = self.abi
        return ret

    def get_arch(self):
        return "riscv"

    def get_backend_config(self, backend):
        if backend in SUPPORTED_TVM_BACKENDS:
            return {
                "target_device": "riscv_cpu",
                "target_march": self.arch,
                "target_model": "unknown",
                "target_mtriple": self.riscv_basename,
                # "target_mattr": "?",
                # "target_mcpu": "?",
                # "target_mabi": self.abi,
            }
        return {}
