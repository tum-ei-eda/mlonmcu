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

import os
import re
import csv
from pathlib import Path

# from mlonmcu.context import MlonMcuContext
from mlonmcu.logging import get_logger

logger = get_logger()

from .common import cli, execute
from .target import Target
from .metrics import Metrics


class RISCVTarget(Target):
    """Common base class for RISCV-like targets. Please do not use this as a target itself!"""

    FEATURES = []

    DEFAULTS = {
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
        return str(self.config["extra_args"])

    @property
    def timeout_sec(self):
        return int(self.config["timeout_sec"])

    def get_target_system(self):
        return "generic_riscv"

    def get_cmake_args(self):
        ret = super().get_cmake_args()
        ret.append(f"-DRISCV_ELF_GCC_PREFIX={self.riscv_prefix}")
        ret.append(f"-DRISCV_ELF_GCC_BASENAME={self.riscv_basename}")
        ret.append(f"-DRISCV_ARCH={self.arch}")
        ret.append(f"-DRISCV_ABI={self.abi}")
        return ret

    def get_arch(self):
        return "riscv"
